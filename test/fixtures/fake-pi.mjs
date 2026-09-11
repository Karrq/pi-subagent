import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

const args = process.argv.slice(2);
const valueAfter = (flag) => {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : undefined;
};

const resume = valueAfter("--session");
const sessionDir = valueAfter("--session-dir");

const emit = (event) => process.stdout.write(`${JSON.stringify(event)}\n`);
const emitUtf8Split = (event) => {
  const encoded = Buffer.from(`${JSON.stringify(event)}\n`);
  const marker = Buffer.from("🙂");
  const markerOffset = encoded.indexOf(marker);
  const splitAt = markerOffset >= 0 ? markerOffset + 2 : Math.floor(encoded.length / 2);
  process.stdout.write(encoded.subarray(0, splitAt));
  process.stdout.write(encoded.subarray(splitAt));
};
const usage = {
  input: 10,
  output: 5,
  cacheRead: 2,
  cacheWrite: 1,
  totalTokens: 18,
  cost: { input: 0.01, output: 0.02, cacheRead: 0.001, cacheWrite: 0.002, total: 0.033 },
};

let sessionFile = resume;
if (!sessionFile) {
  fs.mkdirSync(sessionDir, { recursive: true });
  sessionFile = path.join(sessionDir, "fake-session.jsonl");
  fs.writeFileSync(
    sessionFile,
    `${JSON.stringify({ type: "session", version: 3, id: "fake-session", timestamp: new Date().toISOString(), cwd: process.cwd() })}\n`,
  );
}

// Real RPC mode exits once the parent closes its write end (EOF), which it does right after
// seeing agent_settled -- mirror that instead of exiting on our own.
process.stdin.on("end", () => process.exit(0));

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  if (!line.trim()) return;
  const command = JSON.parse(line);
  if (command.type !== "prompt") return;
  const task = command.message;
  emit({ id: command.id, type: "response", command: "prompt", success: true });

  emit({ type: "message_start", message: { role: "assistant", content: [] } });
  const streamedText = task.includes("hang") ? "partial checkpoint" : task.includes("unicode") ? "नमस्ते🙂" : "finished";
  const update = { type: "message_update", usage, assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: streamedText } };
  if (task.includes("unicode")) emitUtf8Split(update);
  else emit(update);

  if (task.includes("hang")) {
    setInterval(() => {}, 1000);
    return;
  }

  if (task.includes("activity")) {
    emit({ type: "tool_execution_start", toolCallId: "tool-1", toolName: "bash", args: { command: "echo test" } });
    await new Promise((resolve) => setTimeout(resolve, 20));
    emit({ type: "tool_execution_end", toolCallId: "tool-1", toolName: "bash", result: {}, isError: false });
  }
  const message = {
    role: "assistant",
    content: [{ type: "text", text: resume ? "resumed result" : task.includes("unicode") ? streamedText : "finished result" }],
    provider: "test-provider",
    model: "test-model",
    api: "openai-responses",
    usage,
    stopReason: "stop",
    timestamp: Date.now(),
  };
  fs.appendFileSync(sessionFile, `${JSON.stringify({ type: "message", id: "message1", parentId: null, timestamp: new Date().toISOString(), message })}\n`);
  emit({ type: "message_end", message });
  emit({ type: "agent_settled" });
});
