import type { UiLanguage } from "../api/types";
import { t, type TranslationKey } from "../i18n";

export function PlaceholderPage({ page, language }: { page: TranslationKey; language: UiLanguage }) {
  return (
    <main className="page-card placeholder-page">
      <span className="eyebrow">v1.1.0</span>
      <h1>{t(language, page)}</h1>
      <p>{t(language, "pagePending")}</p>
    </main>
  );
}
