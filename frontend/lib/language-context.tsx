"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import { translations, type Language, type TranslationDict } from "@/lib/i18n";

const STORAGE_KEY = "rag-assistant:language";

interface LanguageContextValue {
  language: Language;
  setLanguage: (language: Language) => void;
  t: TranslationDict;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  // Default "en" on the server and on first client render (SSR-safe), then
  // sync from localStorage once mounted.
  const [language, setLanguageState] = useState<Language>("en");

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "en" || stored === "fr") {
        // Deliberate one-time sync from an external, client-only source
        // (localStorage isn't available during SSR/hydration): render the
        // "en" default first so server and client markup match, then read
        // the real preference once mounted. Not a cascading-update risk —
        // this effect has no dependency that this setState could re-trigger.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setLanguageState(stored);
      }
    } catch {
      // localStorage unavailable (private mode, etc.) — stay on the default.
    }
  }, []);

  function setLanguage(next: Language) {
    setLanguageState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Non-fatal: the choice just won't survive a reload.
    }
  }

  return (
    <LanguageContext.Provider
      value={{ language, setLanguage, t: translations[language] }}
    >
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return ctx;
}
