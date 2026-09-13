"use client";

import { Languages } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useLanguage } from "@/lib/language-context";

export function LanguageToggle() {
  const { language, setLanguage, t } = useLanguage();
  const other = language === "en" ? "fr" : "en";

  const label = other.toUpperCase();

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setLanguage(other)}
      // The accessible name must contain the visible label ("EN"/"FR") per
      // WCAG 2.5.3 (Label in Name), so voice-control ("click FR") matches.
      aria-label={`${t.switchLanguage}: ${label}`}
    >
      <Languages />
      {label}
    </Button>
  );
}
