import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { StringDecoder } from "node:string_decoder";
import type { AgentToolResult, ThinkingLevel } from "@earendil-works/pi-agent-core";
import { getSupportedThinkingLevels, StringEnum, type Message, type Model, type Usage } from "@earendil-works/pi-ai";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	defineTool,
	type ExtensionAPI,
	getAgentDir,
	getMarkdownTheme,
	truncateHead,
} from "@earendil-works/pi-coding-agent";
import { Container, Markdown, Spacer, Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";

const CHILD_SYSTEM_GUIDANCE =
	"Work independently on the delegated task. If you cannot proceed reliably without clarification, a decision, or missing information, stop and return the precise question or blocker to your supervisor. Do not guess or wait. Your supervisor can resume this session with an answer.";
const COLLAPSED_ITEM_COUNT = 10;
const CHILD_METADATA_SUFFIX = ".subagent.json";
const INTERNAL_TOOL_NAMES = new Set(["subagent", "subagent_models"]);
const THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const satisfies readonly ThinkingLevel[];
const THINKING_LEVEL_SET = new Set<string>(THINKING_LEVELS);

interface UsageStats extends Usage {
	turns: number;
}

interface ChildMetadata {
	version: 2;
	cwd: string;
	model?: string;
	thinking?: string;
	tools: string[];
	// The label given at creation, if any. Written once and never touched by later resumes, so a
	// resume call without its own label can still show what this subagent was originally called.
	label?: string;
}

interface ToolActivity {
	id: string;
	name: string;
	args: Record<string, unknown>;
	status: "running" | "done" | "error";
}

// Keyed by toolCallId. A running call's own SubagentResult object, mutated in place as the run
// progresses -- absent once the call finishes, since its toolResult then carries the same data.
//
// Pi's extension loader evaluates this file through a fresh jiti instance (moduleCache: false)
// per top-level pi.extensions entry, so subagent-viewer's static import of this module gets its
// own separate copy, not the one this tool's execute() actually mutates. Stashing the Map on
// globalThis under a registry-wide symbol keeps every copy pointing at the same object, since
// globalThis is shared across all jiti-loaded modules in this process.
const ACTIVE_CALLS_KEY = Symbol.for("pi-subagent.activeSubagentCalls");
const globalWithActiveCalls = globalThis as typeof globalThis & { [ACTIVE_CALLS_KEY]?: Map<string, SubagentResult> };
export const activeSubagentCalls: Map<string, SubagentResult> =
	globalWithActiveCalls[ACTIVE_CALLS_KEY] ?? (globalWithActiveCalls[ACTIVE_CALLS_KEY] = new Map());

export interface SubagentResult {
	task: string;
	label?: string;
	resumed: boolean;
	timeoutMs?: number;
	startedAt: number;
	endedAt?: number;
	exitCode: number;
	messages: Message[];
	toolActivity: ToolActivity[];
	partialText: string;
	stderr: string;
	usage: UsageStats;
	// Tokens occupied by the context as of the most recent turn (totalTokens of that turn's
	// usage, not the run's cumulative total): what a resume would actually resend.
	contextUsed?: number;
	contextWindow?: number;
	model?: string;
	thinking?: string;
	stopReason?: string;
	errorMessage?: string;
	sessionFile?: string;
}

export interface SubagentDetails {
	result: SubagentResult;
}

interface ModelAllowlistLevel {
	artificialAnalysis?: { intelligence?: number; coding?: number; cost?: number };
	deepSWE?: { pass?: number; cost?: number };
}

interface ModelAllowlistEntry {
	id: string;
	levels?: Record<string, ModelAllowlistLevel>;
	description?: string;
	[key: string]: unknown;
}

interface ModelAllowlistConfig {
	enabled?: boolean;
	allowed?: (string | ModelAllowlistEntry)[];
	default?: string;
	defaultThinking?: string;
}

interface ModelPolicy {
	enabled: boolean;
	allowed: Set<string>;
	metadata: Map<string, ModelAllowlistEntry>;
	defaultModel?: string;
	defaultThinking?: string;
	configPath: string;
}

interface RunConfig {
	cwd: string;
	model?: string;
	thinking?: string;
	tools: string[];
}

const EMPTY_USAGE = (): UsageStats => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	turns: 0,
});

function getModelAllowlistPath(): string {
	return path.join(getAgentDir(), "pi-subagent", "models-allowlist.json");
}

function loadModelPolicy(configPath = getModelAllowlistPath()): { policy: ModelPolicy; error?: string } {
	const disabled: ModelPolicy = {
		enabled: false,
		allowed: new Set(),
		metadata: new Map(),
		configPath,
	};
	if (!fs.existsSync(configPath)) return { policy: disabled };

	let parsed: unknown;
	try {
		parsed = JSON.parse(fs.readFileSync(configPath, "utf8"));
	} catch (error) {
		return {
			policy: disabled,
			error: `Invalid JSON in model allowlist: ${error instanceof Error ? error.message : String(error)}`,
		};
	}
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		return { policy: disabled, error: "Model allowlist must be a JSON object." };
	}

	const config = parsed as ModelAllowlistConfig;
	if (config.enabled !== undefined && typeof config.enabled !== "boolean") {
		return { policy: disabled, error: 'Model allowlist field "enabled" must be boolean.' };
	}
	if (config.allowed !== undefined && !Array.isArray(config.allowed)) {
		return { policy: disabled, error: 'Model allowlist field "allowed" must be an array.' };
	}
	if (config.default !== undefined && typeof config.default !== "string") {
		return { policy: disabled, error: 'Model allowlist field "default" must be a string.' };
	}
	if (config.defaultThinking !== undefined && typeof config.defaultThinking !== "string") {
		return { policy: disabled, error: 'Model allowlist field "defaultThinking" must be a string.' };
	}
	const defaultThinking = config.defaultThinking?.trim() || undefined;
	if (defaultThinking && !THINKING_LEVEL_SET.has(defaultThinking)) {
		return { policy: disabled, error: `Default thinking level "${defaultThinking}" is not recognized by Pi.` };
	}

	const metadata = new Map<string, ModelAllowlistEntry>();
	const allowed = new Set<string>();
	for (const raw of config.allowed ?? []) {
		if (typeof raw === "string") {
			const id = raw.trim();
			if (id) allowed.add(id);
			continue;
		}
		if (raw && typeof raw === "object" && typeof raw.id === "string" && raw.id.trim()) {
			const entry = { ...raw, id: raw.id.trim() };
			allowed.add(entry.id);
			metadata.set(entry.id, entry);
		}
	}

	const enabled = config.enabled ?? true;
	const defaultModel = config.default?.trim() || undefined;
	if (enabled && allowed.size === 0) {
		return { policy: disabled, error: 'Model allowlist is enabled but "allowed" is empty.' };
	}
	if (enabled && defaultModel && !allowed.has(defaultModel)) {
		return { policy: disabled, error: 'The default model must also appear in "allowed".' };
	}

	return { policy: { enabled, allowed, metadata, defaultModel, defaultThinking, configPath } };
}

function allowedThinkingLevels(entry: ModelAllowlistEntry | undefined): string[] | undefined {
	if (!entry?.levels || typeof entry.levels !== "object" || Array.isArray(entry.levels)) return undefined;
	const levels = Object.keys(entry.levels);
	return levels.length > 0 ? levels : undefined;
}

interface ModelRegistryView {
	getAll(): Model<any>[];
}

function modelMap(registry: ModelRegistryView): Map<string, Model<any>> {
	return new Map(registry.getAll().map((model) => [`${model.provider}/${model.id}`, model]));
}

function effectiveThinkingLevels(entry: ModelAllowlistEntry | undefined, model: Model<any>): ThinkingLevel[] {
	const supported = getSupportedThinkingLevels(model) as ThinkingLevel[];
	const configured = allowedThinkingLevels(entry);
	return configured
		? supported.filter((level) => configured.includes(level))
		: supported;
}

function validateModelPolicy(policy: ModelPolicy, registry: ModelRegistryView): string[] {
	if (!policy.enabled) return [];
	const errors: string[] = [];
	const known = modelMap(registry);

	for (const id of policy.allowed) {
		const model = known.get(id);
		if (!model) {
			errors.push(`Allowlisted model "${id}" is not known to Pi.`);
			continue;
		}
		const supported = new Set<string>(getSupportedThinkingLevels(model));
		for (const level of allowedThinkingLevels(policy.metadata.get(id)) ?? []) {
			if (!THINKING_LEVEL_SET.has(level)) {
				errors.push(`Thinking level "${level}" configured for "${id}" is not recognized by Pi.`);
			} else if (!supported.has(level)) {
				errors.push(`Thinking level "${level}" is unsupported by "${id}".`);
			}
		}
	}
	if (policy.defaultThinking && policy.defaultModel) {
		const defaultModel = known.get(policy.defaultModel);
		if (defaultModel && !effectiveThinkingLevels(policy.metadata.get(policy.defaultModel), defaultModel).includes(policy.defaultThinking as ThinkingLevel)) {
			errors.push(`Default thinking level "${policy.defaultThinking}" is not allowed for "${policy.defaultModel}".`);
		}
	}
	return errors;
}

function resolveFreshModel(
	requestedModel: string | undefined,
	thinking: string | undefined,
	policy: ModelPolicy,
	registry?: ModelRegistryView,
): { model?: string; thinking?: string; error?: string } {
	if (!policy.enabled) return { model: requestedModel?.trim() || undefined };
	const model = requestedModel?.trim() || policy.defaultModel;
	if (!model) return { error: "Model policy requires `model`, but no default is configured." };
	if (!policy.allowed.has(model)) {
		return { error: `Model "${model}" is not allowed. Call subagent_models for permitted models.` };
	}
	const entry = policy.metadata.get(model);
	const configuredLevels = allowedThinkingLevels(entry);
	const knownModel = registry ? modelMap(registry).get(model) : undefined;
	const levels = knownModel ? effectiveThinkingLevels(entry, knownModel) : configuredLevels;
	const selectedThinking = thinking || policy.defaultThinking;
	if (!selectedThinking && configuredLevels) {
		return { error: `Thinking level is required for "${model}". Allowed: ${levels?.join(", ") || "none"}.` };
	}
	if (selectedThinking && levels && !levels.includes(selectedThinking)) {
		return { error: `Thinking level "${selectedThinking}" is not allowed for "${model}". Allowed: ${levels.join(", ") || "none"}.` };
	}
	return selectedThinking && !thinking ? { model, thinking: selectedThinking } : { model };
}

function formatLevels(levels: ThinkingLevel[], metadata: unknown): string {
	const values = metadata && typeof metadata === "object" && !Array.isArray(metadata)
		? metadata as Record<string, ModelAllowlistLevel>
		: {};
	return levels
		.map((level) => {
			const metrics: string[] = [];
			const aa = values[level]?.artificialAnalysis;
			if (aa) {
				const quality = [aa.intelligence, aa.coding]
					.filter((metric): metric is number => typeof metric === "number")
					.map((metric) => metric.toFixed(1))
					.join("/");
				metrics.push(`AA ${quality || "?"}${typeof aa.cost === "number" ? `/$${aa.cost}` : ""}`);
			}
			const swe = values[level]?.deepSWE;
			if (swe) {
				const pass = typeof swe.pass === "number" ? `${Math.round(swe.pass * 100)}%` : "?";
				metrics.push(`DeepSWE ${pass}${typeof swe.cost === "number" ? `/$${swe.cost}` : ""}`);
			}
			return `${level}: ${metrics.join(" · ") || "unbenchmarked"}`;
		})
		.join(" · ");
}

function compactModelCatalog(policy: ModelPolicy, validationErrors: string[], registry: ModelRegistryView) {
	const known = modelMap(registry);
	const entries = Array.from(policy.allowed).map((id) => policy.metadata.get(id) ?? { id });
	return {
		allowlistEnabled: policy.enabled,
		default: policy.defaultModel ?? null,
		defaultThinking: policy.defaultThinking ?? null,
		columns: ["id", "levels", "description"],
		models: entries.map((entry) => {
			const model = known.get(entry.id);
			const levels = model ? effectiveThinkingLevels(entry, model) : [];
			return [entry.id, formatLevels(levels, entry.levels), entry.description ?? null];
		}),
		validationErrors,
		note: policy.enabled
			? "Use a row id as subagent.model and one of its levels as subagent.thinking. Omit model to use the default; omit thinking to use defaultThinking when configured."
			: "No allowlist is configured; omit subagent.model to use the child Pi default or provide any available Pi model.",
	};
}

function metadataPath(sessionFile: string): string {
	return `${sessionFile}${CHILD_METADATA_SUFFIX}`;
}

function writeChildMetadata(sessionFile: string, metadata: ChildMetadata): void {
	const destination = metadataPath(sessionFile);
	const temporary = `${destination}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
	try {
		fs.writeFileSync(temporary, `${JSON.stringify(metadata, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
		fs.renameSync(temporary, destination);
	} finally {
		try {
			if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
		} catch {
			// Preserve the original metadata error.
		}
	}
}

function readChildMetadata(sessionFile: string): ChildMetadata {
	const file = metadataPath(sessionFile);
	let parsed: unknown;
	try {
		parsed = JSON.parse(fs.readFileSync(file, "utf8"));
	} catch {
		throw new Error(`Subagent runtime metadata not found or invalid: ${file}`);
	}
	if (!parsed || typeof parsed !== "object") throw new Error(`Invalid subagent runtime metadata: ${file}`);
	const value = parsed as Partial<ChildMetadata>;
	if (value.version !== 2 || typeof value.cwd !== "string" || !Array.isArray(value.tools)) {
		throw new Error(`Invalid subagent runtime metadata: ${file}`);
	}
	return {
		version: 2,
		cwd: value.cwd,
		model: typeof value.model === "string" ? value.model : undefined,
		thinking: typeof value.thinking === "string" ? value.thinking : undefined,
		tools: value.tools.filter((tool): tool is string => typeof tool === "string"),
		label: typeof value.label === "string" ? value.label : undefined,
	};
}

function expandHome(input: string): string {
	if (input === "~") return os.homedir();
	if (input.startsWith("~/") || input.startsWith("~\\")) return path.join(os.homedir(), input.slice(2));
	return input;
}

function resolveResumePath(input: string): string {
	const expanded = expandHome(input.trim());
	if (!path.isAbsolute(expanded) || path.extname(expanded) !== ".jsonl") {
		throw new Error("`resume` must be the exact absolute JSONL path returned by an earlier subagent call.");
	}
	const resolved = path.resolve(expanded);
	if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
		throw new Error(`Subagent session not found: ${resolved}`);
	}
	return resolved;
}

function resolveSessionFile(sessionDir: string, result: SubagentResult): void {
	if (result.sessionFile) return;
	try {
		const file = fs.readdirSync(sessionDir).find((name) => name.endsWith(".jsonl"));
		if (file) result.sessionFile = path.join(sessionDir, file);
	} catch {
		// The child may not have created its session yet.
	}
}

function getPiInvocation(args: string[]): { command: string; args: string[] } {
	const currentScript = process.argv[1];
	const bunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
	if (currentScript && !bunVirtualScript && fs.existsSync(currentScript)) {
		return { command: process.execPath, args: [currentScript, ...args] };
	}
	const executable = path.basename(process.execPath).toLowerCase();
	if (!/^(node|bun)(\.exe)?$/.test(executable)) return { command: process.execPath, args };
	return { command: "pi", args };
}

function addUsage(total: UsageStats, usage: Usage): void {
	total.input += usage.input || 0;
	total.output += usage.output || 0;
	total.cacheRead += usage.cacheRead || 0;
	total.cacheWrite += usage.cacheWrite || 0;
	total.totalTokens = usage.totalTokens || total.totalTokens;
	total.cost.input += usage.cost?.input || 0;
	total.cost.output += usage.cost?.output || 0;
	total.cost.cacheRead += usage.cost?.cacheRead || 0;
	total.cost.cacheWrite += usage.cost?.cacheWrite || 0;
	total.cost.total += usage.cost?.total || 0;
}

function getFinalOutput(result: SubagentResult): string {
	// A timeout/abort can leave an in-progress turn in partialText after an
	// earlier assistant turn has already completed.
	if (result.partialText.trim()) return result.partialText.trim();
	for (let i = result.messages.length - 1; i >= 0; i--) {
		const message = result.messages[i];
		if (message.role !== "assistant") continue;
		const text = message.content
			.filter((part) => part.type === "text")
			.map((part) => part.text)
			.join("\n")
			.trim();
		if (text) return text;
	}
	return result.partialText.trim();
}

function statusOf(result: SubagentResult): "running" | "done" | "failed" | "timeout" | "aborted" {
	if (result.exitCode === -1) return "running";
	if (result.stopReason === "timeout") return "timeout";
	if (result.stopReason === "aborted") return "aborted";
	if (result.exitCode !== 0 || result.stopReason === "error") return "failed";
	return "done";
}

function resultOutput(result: SubagentResult): string {
	const parts: string[] = [];
	if (result.errorMessage) parts.push(`Error: ${result.errorMessage}`);
	const output = getFinalOutput(result);
	if (output) parts.push(output);
	else if (result.stderr.trim()) parts.push(result.stderr.trim());
	return parts.join("\n\n") || "(no output)";
}

function truncateResult(output: string, sessionFile?: string): string {
	const truncated = truncateHead(output, { maxBytes: DEFAULT_MAX_BYTES, maxLines: DEFAULT_MAX_LINES });
	if (!truncated.truncated) return output;
	const sessionHint = sessionFile
		? ` Read the last assistant message in the child session JSONL at ${sessionFile} if the complete response is needed.`
		: " The complete response may be available in the child session JSONL.";
	return `${truncated.content}\n\n[Output truncated: ${truncated.outputLines} of ${truncated.totalLines} lines, ${truncated.outputBytes} of ${truncated.totalBytes} bytes.]${sessionHint}`;
}

// Mirrors pi's own footer formatting (formatTokens in modes/interactive/components/footer.ts)
// so the reported context figures read the same as the main status bar.
export function formatTokensCompact(count: number): string {
	if (count < 1000) return count.toString();
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
	return `${Math.round(count / 1000000)}M`;
}

function envelope(result: SubagentResult): string {
	const fields: string[] = [];
	if (result.label) fields.push(`label=${result.label}`);
	fields.push(`status=${statusOf(result)}`);
	if (result.contextUsed !== undefined) {
		const context = result.contextWindow
			? `${((result.contextUsed / result.contextWindow) * 100).toFixed(1)}%/${formatTokensCompact(result.contextWindow)}`
			: formatTokensCompact(result.contextUsed);
		fields.push(`context=${context}`);
	}
	if (result.sessionFile) fields.push(`session=${result.sessionFile}`);
	return `[${fields.join(" ")}]`;
}

function modelFacingResult(result: SubagentResult): string {
	return `${envelope(result)}\n${truncateResult(resultOutput(result), result.sessionFile)}`;
}

function killChild(proc: ChildProcess): void {
	if (proc.exitCode !== null || proc.signalCode !== null) return;
	if (process.platform === "win32" && proc.pid) {
		const killer = spawn("taskkill.exe", ["/pid", String(proc.pid), "/t", "/f"], {
			stdio: "ignore",
			windowsHide: true,
		});
		killer.on("error", () => proc.kill());
		killer.on("close", (code) => {
			if (code !== 0 && proc.exitCode === null && proc.signalCode === null) proc.kill();
		});
		killer.unref();
		return;
	}
	proc.kill("SIGTERM");
	setTimeout(() => {
		if (proc.exitCode === null && proc.signalCode === null) proc.kill("SIGKILL");
	}, 5000).unref();
}

type Update = (partial: AgentToolResult<SubagentDetails>) => void;

interface RunChildDependencies {
	agentDir?: string;
	invoke?: (args: string[]) => { command: string; args: string[] };
}

async function runChild(
	config: RunConfig,
	task: string,
	label: string | undefined,
	timeoutMs: number | undefined,
	resumePath: string | undefined,
	signal: AbortSignal | undefined,
	onUpdate: Update | undefined,
	dependencies: RunChildDependencies = {},
	toolCallId?: string,
): Promise<SubagentResult> {
	const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
	const sessionDir = path.join(dependencies.agentDir ?? getAgentDir(), "sessions", "subagent", runId);
	if (!resumePath) await fs.promises.mkdir(sessionDir, { recursive: true });

	const args = ["--mode", "rpc"];
	if (resumePath) args.push("--session", resumePath);
	else args.push("--session-dir", sessionDir);
	if (config.model) args.push("--model", config.model);
	if (config.thinking) args.push("--thinking", config.thinking);
	if (config.tools.length === 0) args.push("--no-tools");
	else args.push("--tools", config.tools.join(","));
	if (config.tools.some((tool) => INTERNAL_TOOL_NAMES.has(tool))) {
		args.push("--extension", path.join(import.meta.dirname, "index.ts"));
	}
	args.push("--append-system-prompt", CHILD_SYSTEM_GUIDANCE);

	const result: SubagentResult = {
		task,
		label,
		resumed: Boolean(resumePath),
		timeoutMs,
		startedAt: Date.now(),
		exitCode: -1,
		messages: [],
		toolActivity: [],
		partialText: "",
		stderr: "",
		usage: EMPTY_USAGE(),
		model: config.model,
		thinking: config.thinking,
		sessionFile: resumePath,
	};

	// Exposed so a viewer extension can inspect in-flight runs, which have no toolResult yet.
	if (toolCallId) activeSubagentCalls.set(toolCallId, result);

	try {
		const emitUpdate = () => {
			onUpdate?.({
				content: [{ type: "text", text: result.partialText || getFinalOutput(result) || "(running...)" }],
				details: { result },
			});
		};

		let aborted = false;
		let timedOut = false;
		let currentTurnUsage: Usage | undefined;

		const exitCode = await new Promise<number>((resolve) => {
			const invocation = (dependencies.invoke ?? getPiInvocation)(args);
			const proc = spawn(invocation.command, invocation.args, {
				cwd: config.cwd,
				shell: false,
				stdio: ["pipe", "pipe", "pipe"],
			});
			let buffer = "";
			const decoder = new StringDecoder("utf8");
			let settled = false;
			let timeout: ReturnType<typeof setTimeout> | undefined;

			const finish = (code: number) => {
				if (settled) return;
				settled = true;
				if (timeout) clearTimeout(timeout);
				if (signal) signal.removeEventListener("abort", onAbort);
				resolve(code);
			};
			const onAbort = () => {
				if (timedOut) return;
				aborted = true;
				if (timeout) clearTimeout(timeout);
				killChild(proc);
			};
			const processLine = (line: string) => {
				if (!line.trim()) return;
				let event: any;
				try {
					event = JSON.parse(line);
				} catch {
					return;
				}

				if (event.type === "response" && event.command === "get_state") {
					// The session is created during the child's startup, before it reads any commands, so
					// asking here always gets a real answer -- no need to guess by polling the directory.
					if (event.success && event.data?.sessionFile) result.sessionFile = event.data.sessionFile;
					emitUpdate();
					return;
				}
				if (event.type === "response" && event.command === "prompt") {
					if (event.success === false) result.errorMessage = event.error;
					return;
				}
				if (event.type === "agent_settled") {
					proc.stdin?.end();
					return;
				}
				if (event.type === "message_start" && event.message?.role === "assistant") {
					result.partialText = "";
					currentTurnUsage = undefined;
					result.usage.turns++;
					return;
				}
				if (event.type === "message_update") {
					if (event.usage) currentTurnUsage = event.usage as Usage;
					const delta = event.assistantMessageEvent;
					if (delta?.type === "text_delta" && typeof delta.delta === "string") {
						result.partialText += delta.delta;
						emitUpdate();
					}
					return;
				}
				if (event.type === "tool_execution_start") {
					result.toolActivity.push({
						id: String(event.toolCallId ?? ""),
						name: String(event.toolName ?? "unknown"),
						args: event.args && typeof event.args === "object" ? event.args : {},
						status: "running",
					});
					emitUpdate();
					return;
				}
				if (event.type === "tool_execution_end") {
					const activity = result.toolActivity.find((item) => item.id === String(event.toolCallId ?? ""));
					if (activity) activity.status = event.isError ? "error" : "done";
					emitUpdate();
					return;
				}
				if (event.type === "message_end" && event.message) {
					const message = event.message as Message;
					result.messages.push(message);
					if (message.role === "assistant") {
						result.partialText = "";
						currentTurnUsage = undefined;
						if (message.usage) {
							addUsage(result.usage, message.usage);
							result.contextUsed = message.usage.totalTokens;
						}
						const reportedModel = message.model as string | undefined;
						const provider = (message as any).provider as string | undefined;
						if (reportedModel && !config.model) {
							result.model = provider ? `${provider}/${reportedModel}` : reportedModel;
							config.model = result.model;
						}
						if (message.stopReason) result.stopReason = message.stopReason;
						if (message.errorMessage) result.errorMessage = message.errorMessage;
					}
					emitUpdate();
					return;
				}
			};

			proc.stdin?.on("error", () => {});
			if (!resumePath) proc.stdin?.write(`${JSON.stringify({ type: "get_state", id: "state" })}\n`);
			proc.stdin?.write(`${JSON.stringify({ type: "prompt", id: "prompt", message: task })}\n`);
			proc.stdout?.on("data", (data) => {
				buffer += decoder.write(data);
				const lines = buffer.split("\n");
				buffer = lines.pop() ?? "";
				for (const line of lines) processLine(line);
			});
			proc.stderr?.on("data", (data) => {
				result.stderr += data.toString();
			});
			proc.on("close", (code) => {
				buffer += decoder.end();
				if (buffer.trim()) processLine(buffer);
				finish(code ?? 0);
			});
			proc.on("error", (error) => {
				result.errorMessage = error.message;
				finish(1);
			});

			if (timeoutMs) {
				timeout = setTimeout(() => {
					if (aborted) return;
					timedOut = true;
					killChild(proc);
				}, timeoutMs);
			}
			if (signal) {
				if (signal.aborted) onAbort();
				else signal.addEventListener("abort", onAbort, { once: true });
			}
		});

		result.exitCode = exitCode;
		if (currentTurnUsage) {
			addUsage(result.usage, currentTurnUsage);
			result.contextUsed = currentTurnUsage.totalTokens;
		}
		if (!resumePath) resolveSessionFile(sessionDir, result);
		if (!resumePath && !result.sessionFile) {
			result.errorMessage = "Child session was not created; this run cannot be resumed.";
			if (!timedOut && !aborted) result.exitCode = 1;
		} else {
			try {
				writeChildMetadata(result.sessionFile!, {
					version: 2,
					cwd: config.cwd,
					model: config.model,
					thinking: config.thinking,
					tools: config.tools,
					label,
				});
			} catch (error) {
				// A fresh child's metadata is load-bearing: it is the only record of cwd/model/
				// thinking/tools, so a failed write means the session can never be resumed. On a
				// resume, the metadata already existed and was already validated; a failed rewrite
				// here just means the label doesn't stick for the next resume.
				if (!resumePath) {
					result.errorMessage = `Child finished, but resume metadata could not be saved: ${error instanceof Error ? error.message : String(error)}`;
					if (!timedOut && !aborted) result.exitCode = 1;
				}
			}
		}
		if (timedOut) result.stopReason = "timeout";
		else if (aborted) result.stopReason = "aborted";
		result.endedAt = Date.now();
		return result;
	} finally {
		if (toolCallId) activeSubagentCalls.delete(toolCallId);
	}
}

function formatTokens(count: number): string {
	if (count < 1000) return String(count);
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	return `${(count / 1000000).toFixed(1)}M`;
}

function formatTimeout(timeoutMs: number | undefined): string {
	if (timeoutMs === undefined) return "";
	if (timeoutMs < 1000) return `${timeoutMs}ms`;
	if (timeoutMs < 60_000) {
		const seconds = Math.round(timeoutMs / 100) / 10;
		return seconds >= 60 ? "1m" : `${seconds}s`;
	}
	if (timeoutMs < 3_600_000) {
		let minutes = Math.floor(timeoutMs / 60_000);
		let seconds = Math.round((timeoutMs % 60_000) / 1000);
		if (seconds === 60) {
			minutes++;
			seconds = 0;
		}
		return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
	}
	let hours = Math.floor(timeoutMs / 3_600_000);
	let minutes = Math.round((timeoutMs % 3_600_000) / 60_000);
	if (minutes === 60) {
		hours++;
		minutes = 0;
	}
	return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
}

function timeoutBadge(timeoutMs: number | undefined): string {
	const formatted = formatTimeout(timeoutMs);
	return formatted ? `[⏱ ${formatted}]` : "";
}

function elapsedBadge(elapsedMs: number): string {
	return `[⏱ ${formatTimeout(Math.max(0, elapsedMs))}]`;
}

function usageText(result: SubagentResult): string {
	const parts: string[] = [];
	if (result.usage.turns) parts.push(`${result.usage.turns} turn${result.usage.turns === 1 ? "" : "s"}`);
	if (result.usage.input) parts.push(`↑${formatTokens(result.usage.input)}`);
	if (result.usage.output) parts.push(`↓${formatTokens(result.usage.output)}`);
	if (result.usage.cacheRead) parts.push(`R${formatTokens(result.usage.cacheRead)}`);
	if (result.usage.cacheWrite) parts.push(`W${formatTokens(result.usage.cacheWrite)}`);
	if (result.usage.cost.total) parts.push(`$${result.usage.cost.total.toFixed(4)}`);
	if (result.usage.totalTokens) parts.push(`ctx:${formatTokens(result.usage.totalTokens)}`);
	if (result.model) parts.push(result.model);
	if (result.thinking) parts.push(result.thinking);
	return parts.join(" ");
}

type DisplayItem =
	| { type: "text"; text: string }
	| { type: "toolCall"; name: string; args: Record<string, unknown> }
	| { type: "toolResult"; name: string; text: string; isError: boolean };

function displayItems(result: SubagentResult): DisplayItem[] {
	const items: DisplayItem[] = [];
	for (const message of result.messages) {
		if (message.role === "assistant") {
			for (const part of message.content) {
				if (part.type === "text" && part.text.trim()) items.push({ type: "text", text: part.text });
				if (part.type === "toolCall") items.push({ type: "toolCall", name: part.name, args: part.arguments });
			}
		} else if (message.role === "toolResult") {
			const text = message.content
				.filter((part) => part.type === "text")
				.map((part) => part.text)
				.join("\n");
			items.push({ type: "toolResult", name: message.toolName, text, isError: message.isError });
		}
	}
	if (result.partialText) items.push({ type: "text", text: result.partialText });
	return items;
}

function formatToolCall(name: string, args: Record<string, unknown>): string {
	if (name === "bash" && typeof args.command === "string") return `$ ${args.command.split("\n")[0]}`;
	const target = args.path ?? args.pattern ?? args.query;
	return target === undefined ? name : `${name} ${String(target)}`;
}

function compactItem(item: DisplayItem): string {
	if (item.type === "toolCall") return `→ ${formatToolCall(item.name, item.args)}`;
	if (item.type === "toolResult") {
		const first = item.text.trim().split("\n")[0] || "(no output)";
		return `← ${item.name}${item.isError ? " failed" : ""}: ${first.slice(0, 160)}`;
	}
	return item.text.split("\n").slice(0, 3).join("\n");
}

const SubagentParams = Type.Object({
	task: Type.String({ minLength: 1, description: "Self-contained assignment for a fresh child, or the next instruction for a resumed child." }),
	label: Type.Optional(Type.String({ minLength: 1, maxLength: 64, pattern: "^[A-Za-z0-9._:-]+$", description: "Short correlation label returned with the result." })),
	model: Type.Optional(Type.String({ minLength: 1, description: "Fresh child only. Pi model pattern or provider/id. With child-model policy enabled, use an exact id from subagent_models; omit to use the policy default, or the child Pi default when policy is disabled." })),
	thinking: Type.Optional(StringEnum(THINKING_LEVELS, { description: "Fresh child only. Pi thinking level supported by the selected model and permitted by child-model policy. subagent_models lists the effective levels; omit to use the child Pi default unless policy requires a level." })),
	tools: Type.Optional(Type.Array(Type.String({ minLength: 1 }), { uniqueItems: true, description: "Fresh child only. Omit to inherit the parent's active tools except subagent and subagent_models; [] disables tools; a non-empty array is the child's exact tool set. Add subagent alongside any other required tools only when the child must delegate further." })),
	cwd: Type.Optional(Type.String({ minLength: 1, description: "Working directory for a fresh child. Defaults to the parent cwd." })),
	timeoutMs: Type.Optional(Type.Integer({ minimum: 1, description: "Supervision checkpoint in milliseconds. On expiry, stops the child and returns any assistant text already emitted plus its resumable session when available." })),
	resume: Type.Optional(Type.String({ minLength: 1, description: "Exact absolute session JSONL path returned by an earlier subagent call. Cannot be combined with fresh-child model, thinking, tools, or cwd." })),
});

const SUBAGENT_GUIDELINES = [
	"Give subagent a self-contained task with relevant context, constraints, and the required output.",
	"Issue independent subagent calls together to run them concurrently; partition write work so children do not edit the same files.",
	"Treat subagent.timeoutMs as a supervision checkpoint, not merely a runtime limit; set it to when control should return for progress review. After a timeout: (1) resume the same session only to request a concise progress report covering completed work, current step, blockers, and proposed next action; do not ask it to continue in that turn; (2) once the report returns, decide whether to resume the child and steer any further work based on that report.",
	"Prefer subagent.resume whenever the child's accumulated context remains useful, including for answers, corrections, or follow-up.",
];

export const __testing = {
	allowedThinkingLevels,
	compactModelCatalog,
	envelope,
	loadModelPolicy,
	metadataPath,
	modelFacingResult,
	readChildMetadata,
	resolveFreshModel,
	runChild,
	statusOf,
	validateModelPolicy,
	writeChildMetadata,
	formatTimeout,
	timeoutBadge,
};

export const subagentModelsToolDefinition = defineTool({
	name: "subagent_models",
	label: "Subagent Models",
	description: "Return the model ids and model-supported thinking levels accepted by subagent under the configured child-model policy, with the default and optional benchmark notes. No child is started.",
	promptSnippet: "List model and thinking values accepted by subagent",
	parameters: Type.Object({}),
	async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
		const { policy, error } = loadModelPolicy();
		if (error) throw new Error(error);
		const validationErrors = validateModelPolicy(policy, ctx.modelRegistry);
		const catalog = compactModelCatalog(policy, validationErrors, ctx.modelRegistry);
		return {
			content: [{ type: "text", text: JSON.stringify(catalog) }],
			details: catalog,
		};
	},
});

// `execute` needs the registering extension's own ExtensionAPI to read the parent's active tools
// (ExtensionContext has no equivalent), so this tool definition is a factory rather than a plain value.
export function createSubagentToolDefinition(pi: ExtensionAPI) {
	return defineTool({
		name: "subagent",
		label: "Subagent",
		description: "Run one task in a separate Pi session and return its final or partial response with a session path that can be resumed.",
		promptSnippet: "Run a task in a separate resumable Pi session",
		promptGuidelines: SUBAGENT_GUIDELINES,
		parameters: SubagentParams,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			if (!params.task.trim()) throw new Error("Subagent task must not be blank.");
			const { policy, error: policyError } = loadModelPolicy();
			if (policyError) throw new Error(policyError);
			const validationErrors = validateModelPolicy(policy, ctx.modelRegistry);
			if (validationErrors.length > 0) throw new Error(validationErrors.join("\n"));

			let resumePath: string | undefined;
			let config: RunConfig;
			// Resuming continues the same agent, so it keeps its original name by default; an explicit
			// label argument still overrides it.
			let label = params.label;
			if (params.resume) {
				if (params.model || params.thinking || params.tools || params.cwd) {
					throw new Error("A resumed subagent accepts only task, resume, label, and timeoutMs. Start a fresh child to change runtime configuration.");
				}
				resumePath = resolveResumePath(params.resume);
				const saved = readChildMetadata(resumePath);
				label ??= saved.label;
				if (policy.enabled && !saved.model) {
					throw new Error("The resumed session does not record a verifiable model. Start a fresh child using the current model policy.");
				}
				if (policy.enabled && saved.model && !policy.allowed.has(saved.model)) {
					throw new Error(`The resumed session uses model "${saved.model}", which is no longer permitted.`);
				}
				const entry = saved.model ? policy.metadata.get(saved.model) : undefined;
				const configuredLevels = allowedThinkingLevels(entry);
				const savedModel = saved.model ? modelMap(ctx.modelRegistry).get(saved.model) : undefined;
				const effectiveLevels = savedModel ? effectiveThinkingLevels(entry, savedModel) : undefined;
				if (policy.enabled && configuredLevels && !saved.thinking) {
					throw new Error(`The resumed session does not record a thinking level required by the current policy for "${saved.model}".`);
				}
				if (policy.enabled && saved.thinking && effectiveLevels && !effectiveLevels.includes(saved.thinking as ThinkingLevel)) {
					throw new Error(`The resumed session uses thinking level "${saved.thinking}", which is no longer permitted or supported for "${saved.model}".`);
				}
				config = saved;
			} else {
				const selected = resolveFreshModel(params.model, params.thinking, policy, ctx.modelRegistry);
				if (selected.error) throw new Error(selected.error);
				const inheritedTools = pi.getActiveTools().filter((name) => !INTERNAL_TOOL_NAMES.has(name));
				config = {
					cwd: path.resolve(params.cwd ?? ctx.cwd),
					model: selected.model,
					thinking: selected.thinking ?? params.thinking,
					tools: params.tools ? [...new Set(params.tools)] : inheritedTools,
				};
			}

			const result = await runChild(
				config,
				params.task,
				label,
				params.timeoutMs,
				resumePath,
				signal,
				onUpdate,
				{},
				toolCallId,
			);
			if (result.model) result.contextWindow = modelMap(ctx.modelRegistry).get(result.model)?.contextWindow;
			return {
				content: [{ type: "text", text: modelFacingResult(result) }],
				details: { result },
				usage: result.usage,
			};
		},
		renderCall(args, theme) {
			const preview = args.task.length > 72 ? `${args.task.slice(0, 72)}…` : args.task;
			const identity = args.label ?? (args.resume ? "resume" : "create");
			const timeout = timeoutBadge(typeof args.timeoutMs === "number" ? args.timeoutMs : undefined);
			return new Text(
				`${theme.fg("toolTitle", theme.bold("subagent "))}${theme.fg("accent", identity)}${timeout ? ` ${theme.fg("dim", timeout)}` : ""}\n  ${theme.fg("dim", preview)}`,
				0,
				0,
			);
		},
		renderResult(toolResult, { expanded }, theme, context) {
			const details = toolResult.details as SubagentDetails | undefined;
			if (!details?.result) {
				const content = toolResult.content[0];
				return new Text(content?.type === "text" ? content.text : "(no output)", 0, 0);
			}
			const result = details.result;
			const status = statusOf(result);
			const success = status === "done";
			const icon = status === "running"
				? theme.fg("warning", "⏳")
				: success
					? theme.fg("success", "✓")
					: theme.fg(status === "timeout" ? "warning" : "error", "✗");
			const title = result.label ?? (result.resumed ? "resume" : "create");
			const items = displayItems(result);

			if (status === "running") {
				if (!context.state.elapsedInterval) context.state.elapsedInterval = setInterval(() => context.invalidate(), 1000);
			} else if (context.state.elapsedInterval) {
				clearInterval(context.state.elapsedInterval);
				context.state.elapsedInterval = undefined;
			}
			const elapsed = elapsedBadge((status === "running" ? Date.now() : (result.endedAt ?? Date.now())) - result.startedAt);

			if (!expanded) {
				const recent = items.slice(-COLLAPSED_ITEM_COUNT);
				let text = `${icon} ${theme.fg("toolTitle", theme.bold(title))} ${theme.fg("dim", elapsed)} ${theme.fg(success ? "success" : "warning", status)}`;
				if (items.length > recent.length) text += `\n${theme.fg("muted", `… ${items.length - recent.length} earlier items`)}`;
				for (const item of recent) text += `\n${theme.fg(item.type === "toolResult" && item.isError ? "error" : "dim", compactItem(item))}`;
				for (const activity of result.toolActivity.filter((item) => item.status === "running")) {
					text += `\n${theme.fg("warning", `⚙ ${formatToolCall(activity.name, activity.args)}`)}`;
				}
				const stats = usageText(result);
				if (stats) text += `\n${theme.fg("dim", stats)}`;
				if (result.sessionFile) text += `\n${theme.fg("dim", `session: ${result.sessionFile}`)}`;
				return new Text(text, 0, 0);
			}

			const container = new Container();
			container.addChild(new Text(`${icon} ${theme.fg("toolTitle", theme.bold(title))} ${theme.fg("dim", elapsed)} ${theme.fg(success ? "success" : "warning", status)}`, 0, 0));
			container.addChild(new Spacer(1));
			container.addChild(new Text(theme.fg("muted", "── Task ──"), 0, 0));
			container.addChild(new Text(result.task, 0, 0));
			container.addChild(new Spacer(1));
			container.addChild(new Text(theme.fg("muted", "── Activity ──"), 0, 0));
			for (const item of items) {
				if (item.type === "text") continue;
				container.addChild(new Text(theme.fg(item.type === "toolResult" && item.isError ? "error" : "dim", compactItem(item)), 0, 0));
			}
			container.addChild(new Spacer(1));
			container.addChild(new Text(theme.fg("muted", "── Result ──"), 0, 0));
			container.addChild(new Markdown(resultOutput(result), 0, 0, getMarkdownTheme()));
			const stats = usageText(result);
			if (stats) container.addChild(new Text(theme.fg("dim", stats), 0, 0));
			if (result.sessionFile) container.addChild(new Text(theme.fg("dim", `session: ${result.sessionFile}`), 0, 0));
			return container;
		},
	});
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(subagentModelsToolDefinition);
	pi.registerTool(createSubagentToolDefinition(pi));
}
