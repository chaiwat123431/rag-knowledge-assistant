"use client";

import { useState } from "react";

import { History, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { type Message } from "@/lib/chat";
import { useConversations } from "@/lib/conversations-context";
import { useLanguage } from "@/lib/language-context";
import { cn } from "cn";

export function ConversationHistory({
  onLoad,
}: {
  onLoad: (messages: Message[]) => void;
}) {
  const { t } = useLanguage();
  const { conversations, activeId, loadConversation, deleteConversation } =
    useConversations();
  const [open, setOpen] = useState(false);

  function handleLoad(id: string) {
    const messages = loadConversation(id);
    if (messages) onLoad(messages);
    setOpen(false);
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm">
          <History />
          {t.history}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-0">
        <p className="px-3 pt-3 pb-1 text-xs font-medium text-muted-foreground">
          {t.savedConversations(conversations.length)}
        </p>
        {conversations.length === 0 ? (
          <p className="px-3 pb-3 text-sm text-muted-foreground">
            {t.noSavedConversations}
          </p>
        ) : (
          <ul className="max-h-80 overflow-y-auto p-1.5 pt-0.5">
            {conversations.map((conversation) => (
              <li key={conversation.id} className="group relative">
                {/* Sibling buttons, not nested: a button-in-a-button is
                    invalid HTML and confusing for screen readers. */}
                <button
                  type="button"
                  onClick={() => handleLoad(conversation.id)}
                  className={cn(
                    "block w-full truncate rounded-md py-1.5 pr-7 pl-2 text-left text-sm hover:bg-muted",
                    conversation.id === activeId && "bg-muted",
                  )}
                >
                  {conversation.title || t.untitledConversation}
                </button>
                <button
                  type="button"
                  aria-label={t.deleteConversation}
                  onClick={() => deleteConversation(conversation.id)}
                  className="absolute top-1/2 right-1 -translate-y-1/2 rounded p-1 text-muted-foreground opacity-0 hover:text-foreground focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100"
                >
                  <X className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </PopoverContent>
    </Popover>
  );
}
