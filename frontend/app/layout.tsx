import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { ConversationsProvider } from "@/lib/conversations-context";
import { LanguageProvider } from "@/lib/language-context";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAG Assistant",
  description: "Ask questions about your documents.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <LanguageProvider>
          <ConversationsProvider>{children}</ConversationsProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
