"use client";

import { useEffect, useRef, useState } from "react";

import { SquarePen } from "lucide-react";

import { ChatComposer } from "@/components/ChatComposer";
import { DocumentUpload } from "@/components/DocumentUpload";
import { LanguageToggle } from "@/components/LanguageToggle";
import { MessageBubble } from "@/components/MessageBubble";
import { Button } from "@/components/ui/button";
import { askQuestion, type HistoryMessage } from "@/lib/api";
import { errorText, type Message } from "@/lib/chat";
import { useLanguage } from "@/lib/language-context";

export default function Home() {
  const { t } = useLanguage();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function newConversation() {
    setMessages([]);
    setInput("");
    setError(null);
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;

    // history = every prior turn; the current question is sent separately
    const history: HistoryMessage[] = messages.map(({ role, content }) => ({
      role,
      content,
    }));

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const res = await askQuestion(question, history);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, citations: res.citations },
      ]);
    } catch (err) {
      setError(errorText(err, t));
      // roll the optimistic question back so a retry starts clean
      setMessages((prev) => prev.slice(0, -1));
      setInput(question);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-1 flex-col">
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <h1 className="text-sm font-semibold">{t.appTitle}</h1>
          <div className="flex items-center gap-1">
            <LanguageToggle />
            <Button
              variant="ghost"
              size="sm"
              onClick={newConversation}
              disabled={loading || messages.length === 0}
            >
              <SquarePen />
              {t.newConversation}
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-6">
        <DocumentUpload />

        <div className="flex flex-1 flex-col gap-5">
          {messages.length === 0 && (
            <p className="text-sm text-muted-foreground">{t.emptyState}</p>
          )}

          {messages.map((message, i) => (
            <MessageBubble key={i} message={message} />
          ))}

          {loading && (
            <p className="text-sm text-muted-foreground">
              {t.thinking} <span className="text-xs">{t.thinkingHint}</span>
            </p>
          )}
          {error && <p className="text-sm text-destructive">{error}</p>}

          <div ref={bottomRef} />
        </div>
      </main>

      <div className="sticky bottom-0 border-t border-border bg-background/80 backdrop-blur">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <ChatComposer
            value={input}
            onChange={setInput}
            onSubmit={handleSend}
            disabled={loading}
          />
        </div>
      </div>
    </div>
  );
}
