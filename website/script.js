"use strict";

// All domain sections remain visible when JavaScript is unavailable.
const tabList = document.querySelector(".application-tabs");
if (tabList) {
  const tabs = [...tabList.querySelectorAll("button")];
  tabList.setAttribute("role", "tablist");
  const activate = (selected, moveFocus = false) => {
    for (const tab of tabs) {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      panel.hidden = !active;
    }
    if (moveFocus) selected.focus();
  };
  for (const [index, tab] of tabs.entries()) {
    tab.setAttribute("role", "tab");
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tab.id);
    panel.tabIndex = 0;
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        activate(tabs[next], true);
      }
    });
  }
  activate(tabs[0]);
  tabList.hidden = false;
}

const copyButton = document.getElementById("copy-citation");
if (copyButton && navigator.clipboard?.writeText) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    const status = document.getElementById("copy-status");
    try {
      await navigator.clipboard.writeText(document.getElementById("bibtex").textContent);
      status.textContent = "BibTeX copied to clipboard.";
    } catch {
      status.textContent = "Select and copy the BibTeX above.";
    }
  });
}
