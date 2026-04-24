import "./globals.css";
import type { Metadata } from "next";
import { Sidebar } from "@/components/scada/Sidebar";
import { Topbar } from "@/components/scada/Topbar";

export const metadata: Metadata = {
  title: "MLAI — Fruit Inspection",
  description: "Edge AI visual inspection for fruit detection & grading",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <div className="flex flex-1 flex-col overflow-hidden">
            <Topbar />
            <main className="flex-1 overflow-y-auto p-6">
              <div className="mx-auto max-w-[1600px]">{children}</div>
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
