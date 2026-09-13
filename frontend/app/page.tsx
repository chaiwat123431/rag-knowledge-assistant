"use client";

import { useEffect, useRef, useState } from "react";

import { SquarePen } from "lucide-react";

import { ChatComposer } from "@/components/ChatComposer";
import { ConversationHistory } from "@/components/ConversationHistory";
import { DocumentUpload } from "@/components/DocumentUpload";
import { LanguageToggle } from "@/components/LanguageToggle";
import { MessageBubble } from "@/components/MessageBubble";
import { Button } from "@/components/ui/button";
import { askQuestion, type HistoryMessage } from "@/lib/api";
import { errorText, type Message } from "@/lib/chat";
import { useConversations } from "@/lib/conversations-context";
import { useLanguage } from "@/lib/language-context";

export default function Home() {
  const { t } = useLanguage();
  const { syncActive, startNew } = useConversations();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function newConversation() {
    // By the time this is clickable (disabled while messages is empty),
    // syncActive has already persisted the current conversation on every
    // message exchanged — nothing left to save, just detach from it.
    setMessages([]);
    setInput("");
    setError(null);
    startNew();
  }

  function handleLoadConversation(loaded: Message[]) {
    setMessages(loaded);
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

    const withQuestion: Message[] = [
      ...messages,
      { role: "user", content: question },
    ];
    setMessages(withQuestion);
    syncActive(withQuestion);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const res = await askQuestion(question, history);
      const withAnswer: Message[] = [
        ...withQuestion,
        { role: "assistant", content: res.answer, citations: res.citations },
      ];
      setMessages(withAnswer);
      syncActive(withAnswer);
    } catch (err) {
      setError(errorText(err, t));
      // roll the optimistic question back so a retry starts clean
      setMessages(messages);
      syncActive(messages);
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
            <ConversationHistory onLoad={handleLoadConversation} />
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
