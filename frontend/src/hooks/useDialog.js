import { useEffect } from "react";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * 模态弹层焦点管理：打开时把焦点移入面板并记录触发元素，Escape 关闭，
 * Tab / Shift+Tab 在面板内循环，关闭或卸载时把焦点还给触发元素。
 * 面板无可聚焦元素时，焦点落到面板容器本身（需可设 tabindex）。
 */
export function useDialog({ open, onClose, panelRef }) {
  useEffect(() => {
    if (!open) return undefined;
    const panel = panelRef?.current || null;
    const trigger = document.activeElement;
    const focusables = () => (panel ? [...panel.querySelectorAll(FOCUSABLE)] : []);
    const first = focusables()[0];
    if (first) {
      first.focus();
    } else if (panel) {
      panel.setAttribute("tabindex", "-1");
      panel.focus();
    }
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose?.();
        return;
      }
      if (event.key !== "Tab" || !panel) return;
      const items = focusables();
      if (items.length === 0) {
        event.preventDefault();
        return;
      }
      const firstEl = items[0];
      const lastEl = items[items.length - 1];
      if (event.shiftKey && document.activeElement === firstEl) {
        event.preventDefault();
        lastEl.focus();
      } else if (!event.shiftKey && document.activeElement === lastEl) {
        event.preventDefault();
        firstEl.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      if (trigger instanceof HTMLElement) trigger.focus();
    };
  }, [open, onClose, panelRef]);
}

export default useDialog;
