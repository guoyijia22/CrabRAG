import { useEffect, useState } from "react";

import type { AppSettings, UiLanguage } from "../api/types";
import { t } from "../i18n";

export type PageId = "chat" | "knowledge" | "graph" | "logs" | "settings" | "evaluation" | "governance";

const pageOrder: PageId[] = ["chat", "knowledge", "graph", "logs", "settings", "evaluation", "governance"];

interface AppHeaderProps {
  settings: AppSettings;
  page: PageId;
  busy: boolean;
  onPageChange: (page: PageId) => void;
  onSettingsChange: (settings: AppSettings) => Promise<void>;
}

export function AppHeader({ settings, page, busy, onPageChange, onSettingsChange }: AppHeaderProps) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(settings.system_name);
  const language = settings.ui_language;

  useEffect(() => setName(settings.system_name), [settings.system_name]);

  async function saveName() {
    if (!editing) {
      setEditing(true);
      return;
    }
    const normalized = name.trim();
    if (normalized.length < 4 || normalized.length > 40) return;
    await onSettingsChange({ ...settings, system_name: normalized });
    setEditing(false);
  }

  async function changeLanguage(next: UiLanguage) {
    if (next !== language) await onSettingsChange({ ...settings, ui_language: next });
  }

  return (
    <header className="topbar">
      <div className="brand-block">
        <span className="app-version">{__CRABRAG_VERSION_LABEL__}</span>
        {editing ? (
          <input aria-label={language === "zh" ? "系统名称" : "System name"} value={name} onChange={(event) => setName(event.target.value)} />
        ) : (
          <strong>{settings.system_name}</strong>
        )}
        <button className="icon-button" type="button" title={editing ? t(language, "saveName") : t(language, "editName")} onClick={saveName} disabled={busy}>
          {editing ? "✓" : "✎"}
        </button>
      </div>
      <nav className="primary-nav" aria-label={language === "zh" ? "主导航" : "Main navigation"}>
        <div className="language-switch" role="group" aria-label={t(language, "language")}>
          <button type="button" className={language === "en" ? "active" : ""} onClick={() => changeLanguage("en")} disabled={busy}>English</button>
          <button type="button" className={language === "zh" ? "active" : ""} onClick={() => changeLanguage("zh")} disabled={busy}>中文</button>
        </div>
        {pageOrder.map((item) => (
          <button key={item} type="button" className={page === item ? "active" : ""} aria-current={page === item ? "page" : undefined} onClick={() => onPageChange(item)}>
            {t(language, item)}
          </button>
        ))}
      </nav>
    </header>
  );
}
