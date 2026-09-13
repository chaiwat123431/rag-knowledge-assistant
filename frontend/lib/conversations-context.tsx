"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { type Message } from "@/lib/chat";
import {
  deriveTitle,
  generateId,
  loadConversations,
  saveConversations,
  sortByRecency,
  MAX_CONVERSATIONS,
  type Conversation,
} from "@/lib/conversations";

interface ConversationsContextValue {
  conversations: Conversation[];
  activeId: string | null;
  /** Upsert (or delete, if `messages` is empty) the active conversation's
   * entry to match `messages`. Call this after every message change that
   * comes from actually chatting (question sent, answer received, a
   * failed send rolled back) — never from just loading a conversation
   * into view, or viewing one would bump its "last activity" for free. */
  syncActive: (messages: Message[]) => void;
  /** Detach from the active conversation (if any) without touching it —
   * by the time this is reachable in the UI, syncActive has already
   * persisted it, so there's nothing left to save here. */
  startNew: () => void;
  /** Marks `id` as active and returns its stored messages for the caller
   * to display, or undefined if it no longer exists. */
  loadConversation: (id: string) => Message[] | undefined;
  deleteConversation: (id: string) => void;
}

const ConversationsContext = createContext<ConversationsContextValue | null>(
  null,
);

export function ConversationsProvider({ children }: { children: ReactNode }) {
  // Empty on the server and on first client render (SSR-safe, same
  // pattern as LanguageProvider), then hydrated from localStorage once
  // mounted.
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  // The source of truth `syncActive` reads for "which id is active": a
  // single handleSend() call invokes it twice (before/after an `await`)
  // through the *same* closure, which still sees `activeId` as it was
  // when that closure was created — a fresh render carrying the id the
  // first call just assigned isn't guaranteed to have landed yet. The ref
  // is written synchronously the moment an id is assigned, so the second
  // call always sees it. (The conversation *list* has the identical
  // problem; that's solved below by always updating it through
  // setConversations's functional form, which React guarantees to run
  // against the latest state regardless of which call scheduled it.)
  const activeIdRef = useRef<string | null>(null);

  useEffect(() => {
    // Deliberate one-time hydration from localStorage, same justified
    // exception as language-context.tsx's identical pattern.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setConversations(loadConversations());
  }, []);

  /** Apply `updater` to the list and persist the result. Always goes
   * through setConversations's functional form so two calls from the same
   * stale closure (see activeIdRef above) still compose correctly instead
   * of the second clobbering the first's update. */
  function update(updater: (prev: Conversation[]) => Conversation[]) {
    setConversations((prev) => {
      const next = updater(prev);
      saveConversations(next);
      return next;
    });
  }

  function setActive(id: string | null) {
    activeIdRef.current = id;
    setActiveId(id);
  }

  function syncActive(messages: Message[]) {
    const id = activeIdRef.current;

    if (messages.length === 0) {
      if (id === null) return; // nothing was ever saved for this draft
      setActive(null);
      update((prev) => prev.filter((c) => c.id !== id));
      return;
    }

    if (id === null) {
      // First message of a fresh conversation -> create its entry.
      const newId = generateId();
      setActive(newId);
      const entry: Conversation = {
        id: newId,
        title: deriveTitle(messages),
        messages,
        updatedAt: Date.now(),
      };
      update((prev) =>
        sortByRecency([entry, ...prev]).slice(0, MAX_CONVERSATIONS),
      );
      return;
    }

    update((prev) =>
      sortByRecency(
        prev.map((c) =>
          c.id === id ? { ...c, messages, updatedAt: Date.now() } : c,
        ),
      ),
    );
  }

  function startNew() {
    setActive(null);
  }

  function loadConversation(id: string): Message[] | undefined {
    const found = conversations.find((c) => c.id === id);
    if (!found) return undefined;
    setActive(id);
    return found.messages;
  }

  function deleteConversation(id: string) {
    if (activeIdRef.current === id) setActive(null);
    update((prev) => prev.filter((c) => c.id !== id));
  }

  return (
    <ConversationsContext.Provider
      value={{
        conversations,
        activeId,
        syncActive,
        startNew,
        loadConversation,
        deleteConversation,
      }}
    >
      {children}
    </ConversationsContext.Provider>
  );
}

export function useConversations(): ConversationsContextValue {
  const ctx = useContext(ConversationsContext);
  if (!ctx) {
    throw new Error(
      "useConversations must be used within a ConversationsProvider",
    );
  }
  return ctx;
}
