"use client";

import { Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useLanguage } from "@/lib/language-context";

interface ChatComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}

export function ChatComposer({
  value,
  onChange,
  onSubmit,
  disabled,
}: ChatComposerProps) {
  const { t } = useLanguage();

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="flex gap-2"
    >
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={t.askPlaceholder}
        className="h-9"
        aria-label={t.questionLabel}
      />
      <Button
        type="submit"
        size="lg"
        disabled={disabled || value.trim() === ""}
        aria-label={t.send}
      >
        <Send />
        {t.send}
      </Button>
    </form>
  );
}
