import * as fs from "node:fs";
import * as path from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
	AssistantMessageComponent,
	BranchSummaryMessageComponent,
	CompactionSummaryMessageComponent,
	CustomMessageComponent,
	type ExtensionAPI,
	type ExtensionCommandContext,
	getMarkdownTheme,
	SessionManager,
	sessionEntryToContextMessages,
	type SessionEntry,
	type SessionInfo,
	type SessionMessageEntry,
	type Theme,
	type ToolDefinition,
	ToolExecutionComponent,
	UserMessageComponent,
} from "@earendil-works/pi-coding-agent";
import { type Component, Container, matchesKey, type OverlayOptions, Spacer, truncateToWidth, type TUI, visibleWidth } from "@earendil-works/pi-tui";
import { activeSubagentCalls, createSubagentToolDefinition, type SubagentDetails, subagentModelsToolDefinition } from "../subagent/index.ts";

// Chrome lines around the scrollable transcript viewport: top rule, header, separator, ...,
// separator, footer, bottom rule.
const VIEWER_CHROME_ROWS = 6;
// Same box chrome as the viewer, plus one for the select list's optional scroll indicator line.
const PICKER_CHROME_ROWS = 7;
const MIN_VIEWPORT_ROWS = 5;
const MIN_LIST_VISIBLE = 5;
// The session list stays compact; it has no reason to grow to the full overlay height.
const MAX_LIST_VISIBLE = 15;
const POLL_INTERVAL_MS = 1000;
const FIRST_MESSAGE_PREVIEW_CHARS = 200;
const MAX_SEARCH_TEXT_CHARS = 20_000;

// Mirrors resolveOverlayLayout's height math in @earendil-works/pi-tui so the overlay's
// self-computed line count never exceeds (and, sized right, exactly matches) what the
// framework will actually allocate. Keep these in sync with OVERLAY_OPTIONS below.
const OVERLAY_MARGIN = { top: 2, bottom: 2, left: 3, right: 3 };
const OVERLAY_MAX_HEIGHT_PERCENT = 70;
const OVERLAY_WIDTH_PERCENT = "90%";

const OVERLAY_OPTIONS: OverlayOptions = {
	width: OVERLAY_WIDTH_PERCENT,
	maxHeight: `${OVERLAY_MAX_HEIGHT_PERCENT}%`,
	margin: OVERLAY_MARGIN,
	anchor: "center",
};

function resolveOverlayContentHeight(terminalRows: number): number {
	const availHeight = Math.max(1, terminalRows - OVERLAY_MARGIN.top - OVERLAY_MARGIN.bottom);
	const requested = Math.floor((terminalRows * OVERLAY_MAX_HEIGHT_PERCENT) / 100);
	return Math.max(1, Math.min(requested, availHeight));
}

function cappedViewportRows(terminalRows: number, chromeRows: number, minRows: number, maxRows: number): number {
	const fromTerminal = resolveOverlayContentHeight(terminalRows) - chromeRows;
	return Math.max(minRows, Math.min(maxRows, fromTerminal));
}

// Pads a line to the overlay's full declared width (so blank/padding regions are covered,
// not just visible text) and wraps the whole thing in a solid background so the panel reads
// as one opaque box instead of letting base conversation content bleed through the gaps.
// Wrapped content (e.g. a SelectList row's own styling) may contain full SGR resets ("\x1b[0m"),
// which would otherwise cancel our background for the remainder of the line — re-assert it
// after every such reset so the fill survives.
function panelLine(theme: Theme, width: number, text: string): string {
	const bg = theme.getBgAnsi("selectedBg");
	const padded = truncateToWidth(sanitizeForDisplay(text), width, "", true).replaceAll("\x1b[0m", `\x1b[0m${bg}`);
	return `${bg}${padded}\x1b[49m`;
}

// Two side walls plus a space of padding inside each.
const BOX_FRAME_COLUMNS = 4;

function boxInnerWidth(width: number): number {
	return Math.max(1, width - BOX_FRAME_COLUMNS);
}

function boxRule(theme: Theme, width: number, left: string, right: string): string {
	const rule = `${left}${"─".repeat(Math.max(0, width - 2))}${right}`;
	return panelLine(theme, width, theme.fg("accent", rule));
}

function boxRow(theme: Theme, width: number, text: string): string {
	const wall = theme.fg("accent", "│");
	const inner = truncateToWidth(sanitizeForDisplay(text), boxInnerWidth(width), "", true);
	return panelLine(theme, width, `${wall} ${inner} ${wall}`);
}

const ESC = "\x1b";
const BEL = "\x07";
const ST = "\u009c";
// Raw C1 string introducers (DCS, SOS, OSC, PM, APC) are aliases some terminals still honour.
const C1_STRING_INTRODUCERS = ["\u0090", "\u0098", "\u009d", "\u009e", "\u009f"];
const SGR_PARAMS = /^[0-9;:]*$/;
// Newlines are dropped rather than honoured: a line that spans rows would shift the whole frame.
const DISPLAY_CONTROL_CHARS = /[\u0000-\u0008\u000a-\u001f\u007f-\u009f]/g;
const TAB_WIDTH = 8;

// Replayed child output carries terminal control sequences verbatim (bash results in particular
// bring OSC 133 shell-integration marks). Emitted mid-overlay they act on the real terminal and
// desync its cursor from the redraw, which ghosts the box. Keep colours, drop everything that
// moves the cursor, erases, or switches terminal modes.
function sanitizeForDisplay(text: string): string {
	if (!text.includes(ESC) && !C1_STRING_INTRODUCERS.some((intro) => text.includes(intro))) {
		return cleanPlainText(text, 0).text;
	}

	let out = "";
	let cursor = 0;
	let column = 0;
	while (cursor < text.length) {
		const at = nextSequenceStart(text, cursor);
		const chunk = cleanPlainText(at === -1 ? text.slice(cursor) : text.slice(cursor, at), column);
		// A carriage return means the terminal would overwrite the row from column 0.
		if (chunk.resetsRow) out = "";
		out += chunk.text;
		column = chunk.column;
		if (at === -1) break;

		if (text[at] === ESC) {
			const seq = readEscapeSequence(text, at);
			if (seq.isSgr) out += text.slice(at, seq.end);
			cursor = seq.end;
		} else {
			const allowBel = text[at] === "\u009d" || text[at] === "\u009f";
			cursor = findStringTerminator(text, at + 1, allowBel);
		}
	}
	return out;
}

function nextSequenceStart(text: string, cursor: number): number {
	let next = text.indexOf(ESC, cursor);
	for (const intro of C1_STRING_INTRODUCERS) {
		const index = text.indexOf(intro, cursor);
		if (index !== -1 && (next === -1 || index < next)) next = index;
	}
	return next;
}

function cleanPlainText(text: string, column: number): { text: string; column: number; resetsRow: boolean } {
	const carriageReturn = text.lastIndexOf("\r");
	const resetsRow = carriageReturn !== -1;
	const clean = (resetsRow ? text.slice(carriageReturn + 1) : text).replace(DISPLAY_CONTROL_CHARS, "");
	let col = resetsRow ? 0 : column;
	if (!clean.includes("\t")) return { text: clean, column: col + visibleWidth(clean), resetsRow };

	// The TUI measures a tab as one cell while terminals advance to the next stop; expand so the
	// measured width matches what gets drawn.
	const parts = clean.split("\t");
	let out = parts[0] ?? "";
	col += visibleWidth(out);
	for (const part of parts.slice(1)) {
		const spaces = TAB_WIDTH - (col % TAB_WIDTH);
		out += " ".repeat(spaces) + part;
		col += spaces + visibleWidth(part);
	}
	return { text: out, column: col, resetsRow };
}

function readEscapeSequence(text: string, start: number): { end: number; isSgr: boolean } {
	const next = text[start + 1];
	if (next === undefined) return { end: text.length, isSgr: false };
	if (next === "[") return readControlSequence(text, start);
	// OSC and APC accept BEL as a terminator in practice; DCS/SOS/PM take ST only.
	if (next === "]" || next === "_") return { end: findStringTerminator(text, start + 2, true), isSgr: false };
	if (next === "P" || next === "X" || next === "^") return { end: findStringTerminator(text, start + 2, false), isSgr: false };
	if (next >= "\u0020" && next <= "\u002f") return { end: Math.min(start + 3, text.length), isSgr: false };
	return { end: start + 2, isSgr: false };
}

// An unterminated string sequence swallows the rest of the line, which is what a terminal does too.
function findStringTerminator(text: string, from: number, allowBel: boolean): number {
	for (let index = from; index < text.length; index++) {
		if (allowBel && text[index] === BEL) return index + 1;
		if (text[index] === ST) return index + 1;
		if (text[index] === ESC && text[index + 1] === "\\") return index + 2;
	}
	return text.length;
}

function readControlSequence(text: string, start: number): { end: number; isSgr: boolean } {
	let index = start + 2;
	const paramStart = index;
	while (index < text.length && text[index]! >= "\u0030" && text[index]! <= "\u003f") index++;
	const paramEnd = index;
	while (index < text.length && text[index]! >= "\u0020" && text[index]! <= "\u002f") index++;
	const hasIntermediates = index > paramEnd;
	const final = text[index];
	if (final === undefined || final < "\u0040" || final > "\u007e") return { end: text.length, isSgr: false };
	return { end: index + 1, isSgr: final === "m" && !hasIntermediates && SGR_PARAMS.test(text.slice(paramStart, paramEnd)) };
}

function flattenMessageText(message: AgentMessage): string {
	if (message.role === "user") {
		return typeof message.content === "string"
			? message.content
			: message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
	}
	if (message.role === "assistant" || message.role === "toolResult") {
		return message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
	}
	return "";
}

interface SubagentSessionInfo extends SessionInfo {
	// One per instruction the parent gave the child: the initial task, plus one per resume.
	userTurns: number;
	toolCalls: number;
}

async function buildSessionInfo(filePath: string): Promise<SubagentSessionInfo | undefined> {
	const stat = await fs.promises.stat(filePath);
	const manager = SessionManager.open(filePath);
	const header = manager.getHeader();
	const messageEntries = manager.getEntries().filter((entry): entry is SessionMessageEntry => entry.type === "message");
	const firstUserEntry = messageEntries.find((entry) => entry.message.role === "user");
	const firstMessage = firstUserEntry ? flattenMessageText(firstUserEntry.message).slice(0, FIRST_MESSAGE_PREVIEW_CHARS) : "";
	const allMessagesText = messageEntries
		.map((entry) => flattenMessageText(entry.message))
		.join("\n")
		.slice(0, MAX_SEARCH_TEXT_CHARS);
	const userTurns = messageEntries.filter((entry) => entry.message.role === "user").length;
	// Counted from the calls themselves, not their results: one assistant message can issue
	// several in parallel, and a call that never returned still happened.
	let toolCalls = 0;
	for (const entry of messageEntries) {
		if (entry.message.role !== "assistant") continue;
		for (const part of entry.message.content) {
			if (part.type === "toolCall") toolCalls++;
		}
	}
	return {
		path: filePath,
		id: header?.id ?? path.basename(filePath, ".jsonl"),
		cwd: header?.cwd ?? "",
		created: header?.timestamp ? new Date(header.timestamp) : stat.birthtime,
		modified: stat.mtime,
		messageCount: messageEntries.length,
		firstMessage,
		allMessagesText,
		userTurns,
		toolCalls,
	};
}

// One invocation of the subagent tool: either the initial call or a resume.
interface SubagentCall {
	toolCallId: string;
	turnId: string;
	turnTimestamp: number;
	sessionFile: string;
	ownLabel?: string;
	resumed: boolean;
	// Still executing: this call has no toolResult yet, only a live entry in activeSubagentCalls.
	running: boolean;
}

// The run directory is named "<epoch>-<random>"; the random half reads better and is still
// unique, so unlabeled calls stay distinguishable from each other instead of colliding on a
// generic word like "resumed child".
function sessionIdentity(session: SubagentSessionInfo): string {
	const runDir = path.basename(path.dirname(session.path));
	const match = runDir.match(/^\d+-([a-z0-9]+)$/i);
	return match?.[1] ?? session.id.slice(0, 8);
}

// An unlabeled call keeps the name established at the session's original invocation (falling
// back to a stable id derived from the session path only if that original call was unlabeled
// too), rather than a generic word that collides across different sessions.
function resolvedLabel(call: SubagentCall, originalCall: SubagentCall, session: SubagentSessionInfo): string {
	return call.ownLabel ?? originalCall.ownLabel ?? sessionIdentity(session);
}

interface SubagentTurnEntry {
	call: SubagentCall;
	session: SubagentSessionInfo;
	// Own label, plus the session's current label in parens when a later call superseded it.
	displayLabel: string;
	// True when a later call resumed this same session under a different label: this row is a
	// historical pointer to where the session ended up, not its current state.
	superseded: boolean;
}

interface SubagentTurn {
	turnId: string;
	timestamp: number;
	entries: SubagentTurnEntry[];
}

async function currentConversationSubagentTurns(ctx: ExtensionCommandContext): Promise<SubagentTurn[]> {
	const entries = ctx.sessionManager.getEntries();

	// Parallel subagent calls share one assistant message; group by that message's id so they
	// land in the same turn/section. The assistant message is persisted as soon as it completes,
	// before any of its tool calls finish executing, so still-running calls are already here too.
	const turnForCall = new Map<string, { turnId: string; timestamp: number }>();
	for (const entry of entries) {
		if (entry.type !== "message" || entry.message.role !== "assistant") continue;
		const timestamp = Date.parse(entry.timestamp);
		for (const part of entry.message.content) {
			if (part.type === "toolCall" && part.name === "subagent") turnForCall.set(part.id, { turnId: entry.id, timestamp });
		}
	}

	const calls: SubagentCall[] = [];
	const finishedToolCallIds = new Set<string>();
	for (const entry of entries) {
		if (entry.type !== "message" || entry.message.role !== "toolResult" || entry.message.toolName !== "subagent") continue;
		finishedToolCallIds.add(entry.message.toolCallId);
		const result = (entry.message.details as SubagentDetails | undefined)?.result;
		if (!result?.sessionFile) continue;
		const turn = turnForCall.get(entry.message.toolCallId) ?? { turnId: entry.id, timestamp: Date.parse(entry.timestamp) };
		calls.push({
			toolCallId: entry.message.toolCallId,
			turnId: turn.turnId,
			turnTimestamp: turn.timestamp,
			sessionFile: result.sessionFile,
			ownLabel: result.label,
			resumed: result.resumed,
			running: false,
		});
	}

	// A call still executing has no toolResult yet, so it never made it into the loop above --
	// pull it from the live registry instead. Appended after every completed call, since a
	// running call is by definition the most recent thing happening to its session file.
	for (const [toolCallId, turn] of turnForCall) {
		if (finishedToolCallIds.has(toolCallId)) continue;
		const live = activeSubagentCalls.get(toolCallId);
		if (!live?.sessionFile) continue;
		calls.push({
			toolCallId,
			turnId: turn.turnId,
			turnTimestamp: turn.timestamp,
			sessionFile: live.sessionFile,
			ownLabel: live.label,
			resumed: live.resumed,
			running: true,
		});
	}

	// Calls are pushed in chronological order, so the first/last write per session file is its
	// original invocation / current state, regardless of how many times it was resumed.
	const originalCallForFile = new Map<string, SubagentCall>();
	const latestCallForFile = new Map<string, SubagentCall>();
	for (const call of calls) {
		if (!originalCallForFile.has(call.sessionFile)) originalCallForFile.set(call.sessionFile, call);
		latestCallForFile.set(call.sessionFile, call);
	}

	const sessionByFile = new Map<string, SubagentSessionInfo>();
	const turns = new Map<string, SubagentTurn>();
	for (const call of calls) {
		let session = sessionByFile.get(call.sessionFile);
		if (session === undefined) {
			try {
				await fs.promises.stat(call.sessionFile);
				const info = await buildSessionInfo(call.sessionFile);
				if (!info) continue;
				session = info;
				sessionByFile.set(call.sessionFile, session);
			} catch {
				continue;
			}
		}

		const originalCall = originalCallForFile.get(call.sessionFile)!;
		const latestCall = latestCallForFile.get(call.sessionFile)!;
		const superseded = latestCall !== call;
		const own = resolvedLabel(call, originalCall, session);
		const displayLabel = superseded ? `${own} (${resolvedLabel(latestCall, originalCall, session)})` : own;

		let turn = turns.get(call.turnId);
		if (!turn) {
			turn = { turnId: call.turnId, timestamp: call.turnTimestamp, entries: [] };
			turns.set(call.turnId, turn);
		}
		turn.entries.push({ call, session, displayLabel, superseded });
	}

	// Map iteration preserves insertion (chronological) order; reverse for newest-first, matching
	// the picker's previous convention.
	return [...turns.values()].reverse();
}

function toolDefinitionsFor(pi: ExtensionAPI): Record<string, ToolDefinition<any, any>> {
	return {
		subagent: createSubagentToolDefinition(pi),
		subagent_models: subagentModelsToolDefinition,
	};
}

class SubagentTranscript {
	readonly container = new Container();
	private readonly renderedEntryIds = new Set<string>();
	private readonly pendingTools = new Map<string, ToolExecutionComponent>();
	private readonly tui: TUI;
	private readonly cwd: string;
	private readonly toolDefinitions: Record<string, ToolDefinition<any, any>>;

	constructor(tui: TUI, cwd: string, toolDefinitions: Record<string, ToolDefinition<any, any>>) {
		this.tui = tui;
		this.cwd = cwd;
		this.toolDefinitions = toolDefinitions;
	}

	appendEntries(entries: SessionEntry[]): boolean {
		let appended = false;
		for (const entry of entries) {
			if (this.renderedEntryIds.has(entry.id)) continue;
			this.renderedEntryIds.add(entry.id);
			for (const message of sessionEntryToContextMessages(entry)) {
				this.appendMessage(message);
				appended = true;
			}
		}
		return appended;
	}

	private appendMessage(message: AgentMessage): void {
		switch (message.role) {
			case "user": {
				const text = flattenMessageText(message);
				if (!text) return;
				if (this.container.children.length > 0) this.container.addChild(new Spacer(1));
				this.container.addChild(new UserMessageComponent(text, getMarkdownTheme()));
				return;
			}
			case "assistant": {
				this.container.addChild(new AssistantMessageComponent(message, false, getMarkdownTheme()));
				for (const part of message.content) {
					if (part.type !== "toolCall") continue;
					const component = new ToolExecutionComponent(
						part.name,
						part.id,
						part.arguments,
						{ showImages: true, imageWidthCells: 60 },
						this.toolDefinitions[part.name],
						this.tui,
						this.cwd,
					);
					component.setArgsComplete();
					component.markExecutionStarted();
					this.container.addChild(component);
					if (message.stopReason === "aborted" || message.stopReason === "error") {
						component.updateResult({
							content: [{ type: "text", text: message.errorMessage || (message.stopReason === "aborted" ? "Aborted" : "Error") }],
							isError: true,
						});
					} else {
						this.pendingTools.set(part.id, component);
					}
				}
				return;
			}
			case "toolResult": {
				const component = this.pendingTools.get(message.toolCallId);
				if (!component) return;
				this.pendingTools.delete(message.toolCallId);
				component.updateResult({ content: message.content, details: message.details, isError: message.isError }, false);
				return;
			}
			case "branchSummary": {
				this.container.addChild(new Spacer(1));
				this.container.addChild(new BranchSummaryMessageComponent(message, getMarkdownTheme()));
				return;
			}
			case "compactionSummary": {
				this.container.addChild(new Spacer(1));
				this.container.addChild(new CompactionSummaryMessageComponent(message, getMarkdownTheme()));
				return;
			}
			case "custom": {
				if (!message.display) return;
				this.container.addChild(new CustomMessageComponent(message, undefined, getMarkdownTheme()));
				return;
			}
			default:
				// bashExecution never occurs in a subagent child (it runs in --mode json -p, without interactive `!` bash).
				return;
		}
	}
}

class SubagentViewerComponent implements Component {
	private readonly session: SessionInfo;
	private readonly tui: TUI;
	private readonly theme: Theme;
	private readonly done: () => void;
	private readonly transcript: SubagentTranscript;
	private scrollTop = 0;
	private followingEnd = true;
	private lastContentHeight = 0;
	private lastViewportHeight = MIN_VIEWPORT_ROWS;
	// Line offset of each user message (the original task, plus one per resume prompt), refreshed
	// every render alongside lastContentHeight/lastViewportHeight so { / } can jump between them.
	private turnOffsets: number[] = [];
	private lastStatSignature: string | undefined;
	private readonly pollTimer: ReturnType<typeof setInterval>;

	constructor(session: SessionInfo, tui: TUI, theme: Theme, toolDefinitions: Record<string, ToolDefinition<any, any>>, done: () => void) {
		this.session = session;
		this.tui = tui;
		this.theme = theme;
		this.done = done;
		this.transcript = new SubagentTranscript(tui, session.cwd || process.cwd(), toolDefinitions);
		this.load();
		this.pollTimer = setInterval(() => this.poll(), POLL_INTERVAL_MS);
	}

	private statSignature(): string | undefined {
		try {
			const stat = fs.statSync(this.session.path);
			return `${stat.size}:${stat.mtimeMs}`;
		} catch {
			return undefined;
		}
	}

	private load(): void {
		try {
			const manager = SessionManager.open(this.session.path);
			this.transcript.appendEntries(manager.buildContextEntries());
		} catch {
			// Leave the transcript empty; render() still shows the header/footer chrome.
		}
		this.lastStatSignature = this.statSignature();
	}

	private poll(): void {
		const signature = this.statSignature();
		if (signature === undefined || signature === this.lastStatSignature) return;
		this.lastStatSignature = signature;
		try {
			const manager = SessionManager.open(this.session.path);
			if (this.transcript.appendEntries(manager.buildContextEntries())) this.tui.requestRender();
		} catch {
			// The child may be mid-write; retry on the next poll.
		}
	}

	dispose(): void {
		clearInterval(this.pollTimer);
	}

	invalidate(): void {
		this.transcript.container.invalidate();
	}

	private scrollBy(delta: number): void {
		const maxScrollTop = Math.max(0, this.lastContentHeight - this.lastViewportHeight);
		const start = this.followingEnd ? maxScrollTop : this.scrollTop;
		this.scrollTop = Math.max(0, Math.min(maxScrollTop, start + delta));
		this.followingEnd = this.scrollTop === maxScrollTop;
		this.tui.requestRender();
	}

	private scrollToStart(): void {
		this.scrollTop = 0;
		this.followingEnd = false;
		this.tui.requestRender();
	}

	private scrollToEnd(): void {
		this.followingEnd = true;
		this.tui.requestRender();
	}

	private jumpToTurn(target: number | undefined): void {
		if (target === undefined) return;
		const maxScrollTop = Math.max(0, this.lastContentHeight - this.lastViewportHeight);
		this.scrollTop = Math.min(target, maxScrollTop);
		this.followingEnd = this.scrollTop >= maxScrollTop;
		this.tui.requestRender();
	}

	private jumpToPrevTurn(): void {
		for (let i = this.turnOffsets.length - 1; i >= 0; i--) {
			if (this.turnOffsets[i]! < this.scrollTop) {
				this.jumpToTurn(this.turnOffsets[i]);
				return;
			}
		}
	}

	private jumpToNextTurn(): void {
		this.jumpToTurn(this.turnOffsets.find((offset) => offset > this.scrollTop));
	}

	// Mirrors Container.render's own loop (concatenate each child's rendered lines in order) but
	// also records where each user message starts, in the same pass -- no extra render work.
	private renderTranscript(width: number): { lines: string[]; turnOffsets: number[] } {
		const lines: string[] = [];
		const turnOffsets: number[] = [];
		for (const child of this.transcript.container.children) {
			if (child instanceof UserMessageComponent) turnOffsets.push(lines.length);
			for (const line of child.render(width)) lines.push(line);
		}
		return { lines, turnOffsets };
	}

	handleInput(data: string): void {
		if (matchesKey(data, "escape") || matchesKey(data, "q") || matchesKey(data, "ctrl+c")) {
			this.done();
			return;
		}
		const page = this.lastViewportHeight;
		const halfPage = Math.max(1, Math.floor(page / 2));
		if (matchesKey(data, "up") || matchesKey(data, "k")) this.scrollBy(-1);
		else if (matchesKey(data, "down") || matchesKey(data, "j")) this.scrollBy(1);
		// Most terminals bind PageUp/PageDown to their own scrollback and never forward them, so
		// ctrl+b/f and ctrl+u/d are the bindings that actually reach us.
		else if (matchesKey(data, "pageUp") || matchesKey(data, "ctrl+b")) this.scrollBy(-page);
		else if (matchesKey(data, "pageDown") || matchesKey(data, "ctrl+f")) this.scrollBy(page);
		else if (matchesKey(data, "ctrl+u")) this.scrollBy(-halfPage);
		else if (matchesKey(data, "ctrl+d")) this.scrollBy(halfPage);
		else if (matchesKey(data, "home")) this.scrollToStart();
		else if (matchesKey(data, "end")) this.scrollToEnd();
		else if (matchesKey(data, "{")) this.jumpToPrevTurn();
		else if (matchesKey(data, "}")) this.jumpToNextTurn();
	}

	render(width: number): string[] {
		const innerWidth = boxInnerWidth(width);
		const viewportHeight = Math.max(
			MIN_VIEWPORT_ROWS,
			resolveOverlayContentHeight(this.tui.terminal.rows) - VIEWER_CHROME_ROWS,
		);
		const { lines: contentLines, turnOffsets } = this.renderTranscript(innerWidth);
		const maxScrollTop = Math.max(0, contentLines.length - viewportHeight);
		this.scrollTop = this.followingEnd ? maxScrollTop : Math.min(this.scrollTop, maxScrollTop);
		this.lastContentHeight = contentLines.length;
		this.lastViewportHeight = viewportHeight;
		this.turnOffsets = turnOffsets;

		const visible = contentLines.slice(this.scrollTop, this.scrollTop + viewportHeight);
		while (visible.length < viewportHeight) visible.push("");

		const position = maxScrollTop > 0
			? this.theme.fg("muted", ` (${this.scrollTop + 1}-${Math.min(this.scrollTop + viewportHeight, contentLines.length)}/${contentLines.length})`)
			: "";
		const header = `${this.theme.fg("toolTitle", this.theme.bold("Subagent transcript"))} ${this.theme.fg("dim", this.session.cwd || this.session.path)}${position}`;
		const footer = [
			["↑/↓ j/k", "scroll"],
			["ctrl+u/d", "half page"],
			["ctrl+b/f", "page"],
			["{ / }", "prev/next turn"],
			["home/end", "top/bottom"],
			["esc/q", "close"],
		].map(([key, label]) => `${this.theme.fg("dim", key!)} ${this.theme.fg("muted", label!)}`).join("   ");

		return [
			boxRule(this.theme, width, "╭", "╮"),
			boxRow(this.theme, width, header),
			boxRule(this.theme, width, "├", "┤"),
			...visible.map((line) => boxRow(this.theme, width, line)),
			boxRule(this.theme, width, "├", "┤"),
			boxRow(this.theme, width, footer),
			boxRule(this.theme, width, "╰", "╯"),
		];
	}
}

// One flattened row in the picker: a non-selectable turn divider, or a selectable session call.
type PickerRow = { kind: "header"; timestamp: number } | { kind: "item"; entry: SubagentTurnEntry };

const PRIMARY_COLUMN_MIN = 20;
const PRIMARY_COLUMN_MAX = 40;
const PRIMARY_COLUMN_GAP = 2;
const MIN_DESCRIPTION_WIDTH = 10;

class SubagentPickerComponent implements Component {
	private readonly rows: PickerRow[];
	private readonly itemRowIndices: number[];
	private readonly listVisible: number;
	private readonly theme: Theme;
	private readonly done: (result: SubagentSessionInfo | undefined) => void;
	private selectedPos = 0;

	constructor(turns: SubagentTurn[], tui: TUI, theme: Theme, done: (result: SubagentSessionInfo | undefined) => void) {
		this.theme = theme;
		this.done = done;
		this.rows = turns.flatMap((turn): PickerRow[] => [
			{ kind: "header", timestamp: turn.timestamp },
			...turn.entries.map((entry): PickerRow => ({ kind: "item", entry })),
		]);
		this.itemRowIndices = this.rows.reduce<number[]>((acc, row, i) => {
			if (row.kind === "item") acc.push(i);
			return acc;
		}, []);
		this.listVisible = cappedViewportRows(tui.terminal.rows, PICKER_CHROME_ROWS, MIN_LIST_VISIBLE, MAX_LIST_VISIBLE);
	}

	invalidate(): void {}

	handleInput(data: string): void {
		if (this.itemRowIndices.length === 0) return;
		const last = this.itemRowIndices.length - 1;
		if (matchesKey(data, "up") || matchesKey(data, "k")) this.selectedPos = this.selectedPos === 0 ? last : this.selectedPos - 1;
		else if (matchesKey(data, "down") || matchesKey(data, "j")) this.selectedPos = this.selectedPos === last ? 0 : this.selectedPos + 1;
		else if (matchesKey(data, "enter") || matchesKey(data, "return")) {
			const row = this.rows[this.itemRowIndices[this.selectedPos]!];
			if (row?.kind === "item") this.done(row.entry.session);
		} else if (matchesKey(data, "escape") || matchesKey(data, "q") || matchesKey(data, "ctrl+c")) this.done(undefined);
	}

	private primaryColumnWidth(): number {
		const widest = this.rows.reduce(
			(max, row) => (row.kind === "item" ? Math.max(max, visibleWidth(row.entry.call.running ? `${row.entry.displayLabel} ⏳` : row.entry.displayLabel)) : max),
			0,
		);
		return Math.max(PRIMARY_COLUMN_MIN, Math.min(PRIMARY_COLUMN_MAX, widest + PRIMARY_COLUMN_GAP));
	}

	private renderRow(row: PickerRow, width: number, isSelected: boolean, primaryColumnWidth: number): string {
		if (row.kind === "header") return this.theme.fg("dim", this.theme.bold(new Date(row.timestamp).toLocaleString()));

		const prefix = isSelected ? "→ " : "  ";
		const prefixWidth = visibleWidth(prefix);
		// A running call can't have been superseded yet -- nothing later has happened to it.
		const rawLabel = row.entry.call.running ? `${row.entry.displayLabel} ⏳` : row.entry.displayLabel;
		const dim = row.entry.superseded;
		// Selection only highlights the label; the description (cwd, turns, tool calls) stays
		// visible either way so it isn't hidden the moment you select a row.
		const colorLabel = (text: string) => (isSelected ? this.theme.fg("accent", text) : dim ? this.theme.fg("dim", text) : text);
		const description = `${row.entry.session.cwd || "?"} · ${row.entry.session.userTurns} turn${row.entry.session.userTurns === 1 ? "" : "s"} · ${row.entry.session.toolCalls} tool call${row.entry.session.toolCalls === 1 ? "" : "s"}`;
		if (width <= 40) {
			const label = truncateToWidth(rawLabel, width - prefixWidth, "");
			return `${prefix}${colorLabel(label)}`;
		}
		const maxLabelWidth = Math.max(1, primaryColumnWidth - PRIMARY_COLUMN_GAP);
		const label = truncateToWidth(rawLabel, maxLabelWidth, "");
		const spacing = " ".repeat(Math.max(1, primaryColumnWidth - visibleWidth(label)));
		const remaining = width - prefixWidth - primaryColumnWidth - 2;
		if (remaining <= MIN_DESCRIPTION_WIDTH) return `${prefix}${colorLabel(label)}`;
		const descText = this.theme.fg(dim ? "dim" : "muted", truncateToWidth(description, remaining, ""));
		return `${prefix}${colorLabel(label)}${spacing}${descText}`;
	}

	render(width: number): string[] {
		const inner = boxInnerWidth(width);
		const header = this.theme.fg("toolTitle", this.theme.bold("Subagent sessions"));
		const footer = `${this.theme.fg("dim", "↑/↓ j/k")} ${this.theme.fg("muted", "select")}   ${this.theme.fg("dim", "enter")} ${this.theme.fg("muted", "open")}   ${this.theme.fg("dim", "esc")} ${this.theme.fg("muted", "close")}`;

		let body: string[];
		if (this.rows.length === 0) {
			body = [this.theme.fg("muted", "No subagent sessions found yet.")];
		} else {
			const selectedRawIndex = this.itemRowIndices[this.selectedPos] ?? 0;
			const startIndex = Math.max(0, Math.min(selectedRawIndex - Math.floor(this.listVisible / 2), this.rows.length - this.listVisible));
			const endIndex = Math.min(startIndex + this.listVisible, this.rows.length);
			const primaryColumnWidth = this.primaryColumnWidth();
			body = this.rows
				.slice(startIndex, endIndex)
				.map((row, i) => this.renderRow(row, inner, startIndex + i === selectedRawIndex, primaryColumnWidth));
			if (startIndex > 0 || endIndex < this.rows.length) {
				body.push(this.theme.fg("muted", `  (${this.selectedPos + 1}/${this.itemRowIndices.length})`));
			}
		}

		return [
			boxRule(this.theme, width, "╭", "╮"),
			boxRow(this.theme, width, header),
			boxRule(this.theme, width, "├", "┤"),
			...body.map((line) => boxRow(this.theme, width, line)),
			boxRule(this.theme, width, "├", "┤"),
			boxRow(this.theme, width, footer),
			boxRule(this.theme, width, "╰", "╯"),
		];
	}
}

export const __testing = {
	resolveOverlayContentHeight,
	cappedViewportRows,
	MAX_LIST_VISIBLE,
	sanitizeForDisplay,
	currentConversationSubagentTurns,
	boxInnerWidth,
	boxRow,
	boxRule,
	OVERLAY_MARGIN,
	OVERLAY_MAX_HEIGHT_PERCENT,
	VIEWER_CHROME_ROWS,
	PICKER_CHROME_ROWS,
	MIN_VIEWPORT_ROWS,
	MIN_LIST_VISIBLE,
};

export default function (pi: ExtensionAPI) {
	pi.registerCommand("subagents", {
		description: "Browse subagent child conversations (read-only)",
		async handler(_args, ctx) {
			const turns = await currentConversationSubagentTurns(ctx);
			if (turns.length === 0) {
				ctx.ui.notify("No subagent sessions found yet.", "info");
				return;
			}

			const selected = await ctx.ui.custom<SessionInfo | undefined>(
				(tui, theme, _keybindings, done) => new SubagentPickerComponent(turns, tui, theme, done),
				{ overlay: true, overlayOptions: OVERLAY_OPTIONS },
			);
			if (!selected) return;

			const toolDefinitions = toolDefinitionsFor(pi);
			await ctx.ui.custom<void>(
				(tui, theme, _keybindings, done) => new SubagentViewerComponent(selected, tui, theme, toolDefinitions, () => done()),
				{ overlay: true, overlayOptions: OVERLAY_OPTIONS },
			);
		},
	});
}
