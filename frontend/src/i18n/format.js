import { IntlMessageFormat } from "intl-messageformat";
import en from "./en.json" with { type: "json" };
import zh from "./zh.json" with { type: "json" };

function flatten(object, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(object)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) flatten(value, path, out);
    else out[path] = value;
  }
  return out;
}

const FLAT = { en: flatten(en), zh: flatten(zh) };
const FORMATTERS = new Map();

export function localeOf(language) {
  return language === "en" ? "en" : "zh";
}

export function catalogKeys(language) {
  return Object.keys(FLAT[localeOf(language)]).sort();
}

export function t(language, key, params) {
  const locale = localeOf(language);
  const message = FLAT[locale][key] ?? FLAT.en[key];
  if (message == null) return key;
  if (!params || Object.keys(params).length === 0) return String(message);
  const cacheKey = `${locale}:${key}`;
  let formatter = FORMATTERS.get(cacheKey);
  if (!formatter) {
    formatter = new IntlMessageFormat(message, locale === "en" ? "en" : "zh-Hans");
    FORMATTERS.set(cacheKey, formatter);
  }
  return formatter.format(params);
}

export function issueError(code, params) {
  const error = new Error(code);
  error.issue = params ? { code, params } : { code };
  return error;
}

export function formatIssue(language, issue) {
  if (issue == null || issue === "") return "";
  if (Array.isArray(issue)) return formatIssues(language, issue);
  if (typeof issue === "string") {
    return FLAT.en[issue] || FLAT.zh[issue] ? t(language, issue) : issue;
  }
  if (typeof issue !== "object") return String(issue);
  if (issue.issues) return formatIssues(language, issue.issues);
  if (issue.issue) return formatIssue(language, issue.issue);
  const params = { ...(issue.params || {}) };
  if (issue.inner) params.message = formatIssue(language, issue.inner);
  if (issue.code) return t(language, issue.code, params);
  if (issue.message != null) return formatIssue(language, issue.message);
  return String(issue);
}

export function formatIssues(language, issues, separator) {
  const sep = separator ?? (language === "en" ? "; " : "；");
  return (issues || []).map((issue) => formatIssue(language, issue)).join(sep);
}

export function issueKey(issue, index = 0) {
  if (issue && typeof issue === "object" && issue.code) {
    return `${issue.code}:${JSON.stringify(issue.params || {})}:${index}`;
  }
  return String(issue ?? index);
}
