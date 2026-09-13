"use client";

import { Citations } from "@/components/Citations";
import { type Message } from "@/lib/chat";
import { useLanguage } from "@/lib/language-context";
import { cn } from "cn";

export function MessageBubble({ message }: { message: Message }) {
  const { t } = useLanguage();
  const isUser = message.role === "user";

  return (
    <div className={cn("flex flex-col gap-1", isUser && "items-end")}>
      <span className="text-xs font-medium text-muted-foreground">
        {isUser ? t.you : t.assistant}
      </span>
      <div
        className={cn(
          "text-sm whitespace-pre-wrap",
          isUser
            ? "max-w-[85%] rounded-2xl bg-muted px-4 py-2"
            : "w-full leading-relaxed",
        )}
      >
        {message.content}
      </div>

      {!isUser && message.citations && (
        <Citations citations={message.citations} />
      )}
    </div>
  );
}
