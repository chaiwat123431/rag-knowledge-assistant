// Pure data logic for saved conversations — no React here, mirroring how
// lib/i18n.ts is plain data separate from language-context.tsx's wiring.

import { type Message } from "@/lib/chat";

export const STORAGE_KEY = "rag-assistant:conversations";

// Keeps localStorage usage bounded for a single-user local tool; beyond
// this, the oldest (by last activity) are dropped.
export const MAX_CONVERSATIONS = 50;

const TITLE_MAX_LENGTH = 60;

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

export function generateId(): string {
  return crypto.randomUUID();
}

/**
 * Title from the first user message, whitespace-collapsed and truncated
 * at a word boundary where reasonable. Only ever called right after a
 * conversation gets its first message (see ConversationsProvider), so a
 * user message is always present in practice — the fallback is a plain
 * string, not routed through i18n, since threading translations into this
 * data-only module for a defensive case that shouldn't trigger isn't
 * worth the coupling.
 */
export function deriveTitle(messages: Message[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  const raw = (firstUser?.content ?? "Untitled").replace(/\s+/g, " ").trim();
  if (raw.length <= TITLE_MAX_LENGTH) return raw;

  const truncated = raw.slice(0, TITLE_MAX_LENGTH);
  const lastSpace = truncated.lastIndexOf(" ");
  const clean =
    lastSpace > TITLE_MAX_LENGTH * 0.6 ? truncated.slice(0, lastSpace) : truncated;
  return `${clean}…`;
}

export function sortByRecency(list: Conversation[]): Conversation[] {
  return [...list].sort((a, b) => b.updatedAt - a.updatedAt);
}

function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.title === "string" &&
    typeof v.updatedAt === "number" &&
    Array.isArray(v.messages)
  );
}

/** Reads and validates the saved list; never throws. */
export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Drop individually malformed entries rather than discarding the
    // whole list (and thus every other conversation) over one bad record.
    return sortByRecency(parsed.filter(isConversation));
  } catch {
    return [];
  }
}

/** Writes the list; non-fatal on failure (quota exceeded, private mode,
 * localStorage disabled, ...) — the app keeps working in-memory for the
 * session, it just won't survive a reload. Same trade-off as the language
 * preference in language-context.tsx. */
export function saveConversations(list: Conversation[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    // ignore
  }
}
