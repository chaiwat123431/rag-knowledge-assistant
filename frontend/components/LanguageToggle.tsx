"use client";

import { Languages } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useLanguage } from "@/lib/language-context";

export function LanguageToggle() {
  const { language, setLanguage, t } = useLanguage();
  const other = language === "en" ? "fr" : "en";

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setLanguage(other)}
      aria-label={t.switchLanguage}
    >
      <Languages />
      {other.toUpperCase()}
    </Button>
  );
}
