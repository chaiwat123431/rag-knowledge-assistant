"use client";

import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  askQuestion,
  ingestDocument,
  type Citation,
  type HistoryMessage,
  type Role,
} from "@/lib/api";

interface ChatMessage {
  role: Role;
  content: string;
  citations?: Citation[];
}

type UploadState = "idle" | "uploading" | "done" | "error";

function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    const where = err.status ? `HTTP ${err.status}` : "network error";
    return `${where}: ${err.message}`;
  }
  return "Something went wrong.";
}

export default function Home() {
  // --- upload state ---
  const [file, setFile] = useState<File | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- chat state (messages doubles as the history sent to the backend) ---
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  async function handleUpload() {
    if (!file || uploadState === "uploading") return;
    setUploadState("uploading");
    setUploadMessage(`Uploading ${file.name}…`);
    try {
      const res = await ingestDocument(file);
      setUploadState("done");
      setUploadMessage(
        `Indexed "${res.source}" — ${res.chunks_added} chunk${
          res.chunks_added === 1 ? "" : "s"
        } added.`,
      );
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setUploadState("error");
      setUploadMessage(`Upload failed — ${errorText(err)}`);
    }
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || chatLoading) return;

    // history = every turn so far, current question sent separately
    const history: HistoryMessage[] = messages.map(({ role, content }) => ({
      role,
      content,
    }));

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setChatError(null);
    setChatLoading(true);

    try {
      const res = await askQuestion(question, history);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, citations: res.citations },
      ]);
    } catch (err) {
      setChatError(errorText(err));
      // roll the optimistic question back so a retry starts clean
      setMessages((prev) => prev.slice(0, -1));
      setInput(question);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-4 py-10">
      <h1 className="text-2xl font-semibold">RAG Assistant</h1>

      {/* --- upload --- */}
      <section className="rounded-lg border border-black/10 p-4 dark:border-white/15">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-zinc-500">
          Add a document
        </h2>
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setUploadState("idle");
              setUploadMessage("");
            }}
            className="text-sm"
          />
          <button
            type="button"
            onClick={handleUpload}
            disabled={!file || uploadState === "uploading"}
            className="rounded-md bg-zinc-900 px-3 py-1.5 text-sm text-white disabled:opacity-40 dark:bg-white dark:text-zinc-900"
          >
            {uploadState === "uploading" ? "Uploading…" : "Upload"}
          </button>
        </div>
        {uploadMessage && (
          <p
            className={`mt-3 text-sm ${
              uploadState === "error" ? "text-red-600" : "text-zinc-600 dark:text-zinc-400"
            }`}
          >
            {uploadMessage}
          </p>
        )}
      </section>

      {/* --- chat --- */}
      <section className="flex flex-1 flex-col gap-4">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
          Ask a question
        </h2>

        <div className="flex flex-col gap-4">
          {messages.length === 0 && (
            <p className="text-sm text-zinc-500">
              No messages yet. Upload a document, then ask something about it.
            </p>
          )}

          {messages.map((message, i) => (
            <div key={i} className="flex flex-col gap-1">
              <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                {message.role === "user" ? "You" : "Assistant"}
              </span>
              <p className="whitespace-pre-wrap text-sm">{message.content}</p>

              {message.citations && message.citations.length > 0 && (
                <ul className="mt-2 flex flex-col gap-2">
                  {message.citations.map((citation) => (
                    <li
                      key={citation.marker}
                      className="rounded-md border border-black/10 bg-black/[0.02] p-2 text-xs dark:border-white/15 dark:bg-white/[0.03]"
                    >
                      <span className="font-medium">
                        [{citation.marker}] {citation.source ?? "unknown"}
                      </span>
                      <p className="mt-1 whitespace-pre-wrap text-zinc-600 dark:text-zinc-400">
                        {citation.text}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}

          {chatLoading && (
            <p className="text-sm text-zinc-500">Thinking… (the local model can take a few seconds)</p>
          )}
          {chatError && <p className="text-sm text-red-600">{chatError}</p>}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSend} className="mt-2 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={chatLoading}
            placeholder="Ask about your documents…"
            className="flex-1 rounded-md border border-black/15 bg-transparent px-3 py-2 text-sm outline-none focus:border-black/40 disabled:opacity-50 dark:border-white/20 dark:focus:border-white/50"
          />
          <button
            type="submit"
            disabled={chatLoading || input.trim() === ""}
            className="rounded-md bg-zinc-900 px-4 py-2 text-sm text-white disabled:opacity-40 dark:bg-white dark:text-zinc-900"
          >
            Send
          </button>
        </form>
      </section>
    </main>
  );
}
