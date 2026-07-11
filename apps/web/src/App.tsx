import { useEffect, useState } from "react";

import { getAppSettings, putAppSettings } from "./api/client";
import type { AppSettings } from "./api/types";
import { AppHeader, type PageId } from "./components/AppHeader";
import { ChatPage } from "./pages/ChatPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { GraphPage } from "./pages/GraphPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { LogsPage } from "./pages/LogsPage";
import { SettingsPage } from "./pages/SettingsPage";

export const DEFAULT_APP_SETTINGS: AppSettings = {
  system_name: "CrabRAG",
  knowledge_base_name: "通用基础查询知识库",
  ui_theme: "red_white",
  ui_language: "en",
  sidebar_image_url: "",
  knowledge_base_dirs: [],
  common_questions: [],
  business_scope_description: "General knowledge base assistant for local documents.",
  in_scope_keywords: [],
  out_of_scope_keywords: ["股票", "Stock"],
  scope_min_score: 0,
  out_of_scope_response: "当前问题不属于本系统配置的查询范围，无法为您解答。",
  no_match_response: "暂无相关知识库依据，无法为您解答",
};

export function App() {
  const [page, setPage] = useState<PageId>("chat");
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [bootError, setBootError] = useState("");

  useEffect(() => {
    let active = true;
    getAppSettings()
      .then((payload) => active && setSettings(payload))
      .catch((reason: unknown) => active && setBootError(reason instanceof Error ? reason.message : String(reason)));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    document.title = settings.system_name;
    document.documentElement.lang = settings.ui_language === "zh" ? "zh-CN" : "en";
  }, [settings.system_name, settings.ui_language]);

  async function persistAppSettings(next: AppSettings) {
    const previous = settings;
    setSettings(next);
    setSettingsBusy(true);
    setBootError("");
    try {
      const saved = await putAppSettings(next);
      setSettings({ ...saved, ui_language: next.ui_language });
    } catch (reason) {
      setSettings(previous);
      setBootError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSettingsBusy(false);
    }
  }

  return (
    <div className="app" data-theme={settings.ui_theme} data-language={settings.ui_language}>
      <AppHeader settings={settings} page={page} busy={settingsBusy} onPageChange={setPage} onSettingsChange={persistAppSettings} />
      {bootError && <div className="global-alert alert error" role="alert">{bootError}</div>}
      {page === "chat" && <ChatPage settings={settings} onSettingsChange={setSettings} />}
      {page === "knowledge" && <KnowledgePage language={settings.ui_language} />}
      {page === "governance" && <KnowledgePage language={settings.ui_language} governanceOnly />}
      {page === "graph" && <GraphPage language={settings.ui_language} />}
      {page === "logs" && <LogsPage language={settings.ui_language} />}
      {page === "evaluation" && <EvaluationPage language={settings.ui_language} />}
      {page === "settings" && <SettingsPage language={settings.ui_language} onAppSettingsSaved={setSettings} />}
    </div>
  );
}
