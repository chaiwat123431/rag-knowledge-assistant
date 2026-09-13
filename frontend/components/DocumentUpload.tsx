"use client";

import { useRef, useState } from "react";

import { Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ingestDocument } from "@/lib/api";
import { errorText } from "@/lib/chat";
import { useLanguage } from "@/lib/language-context";
import { cn } from "cn";

type UploadState = "idle" | "uploading" | "done" | "error";

export function DocumentUpload() {
  const { t } = useLanguage();
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [message, setMessage] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload() {
    if (!file || state === "uploading") return;
    setState("uploading");
    setMessage(t.uploadingFile(file.name));
    try {
      const res = await ingestDocument(file);
      setState("done");
      setMessage(t.uploadSuccess(res.source, res.chunks_added));
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      setState("error");
      setMessage(t.uploadFailed(errorText(err, t)));
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t.addDocument}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <Input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,.md"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setState("idle");
              setMessage("");
            }}
            className="flex-1"
          />
          <Button
            type="button"
            variant="outline"
            onClick={handleUpload}
            disabled={!file || state === "uploading"}
          >
            <Upload />
            {state === "uploading" ? t.uploading : t.upload}
          </Button>
        </div>
        {message && (
          <p
            className={cn(
              "text-sm",
              state === "error" ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
