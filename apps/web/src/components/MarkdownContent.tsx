import type { ReactNode } from "react";

function inlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    return part;
  });
}

export function MarkdownContent({ value }: { value: string }) {
  const lines = value.replace(/\r\n/g, "\n").split("\n");
  return (
    <div className="markdown-content">
      {lines.map((line, index) => {
        if (line.startsWith("### ")) return <h4 key={index}>{inlineMarkdown(line.slice(4))}</h4>;
        if (line.startsWith("## ")) return <h3 key={index}>{inlineMarkdown(line.slice(3))}</h3>;
        if (line.startsWith("# ")) return <h2 key={index}>{inlineMarkdown(line.slice(2))}</h2>;
        if (/^[-*] /.test(line)) return <div className="markdown-list-item" key={index}>• {inlineMarkdown(line.slice(2))}</div>;
        return line ? <p key={index}>{inlineMarkdown(line)}</p> : <br key={index} />;
      })}
    </div>
  );
}
