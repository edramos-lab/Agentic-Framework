import React from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import NewRun from "./pages/NewRun";
import RunDetail from "./pages/RunDetail";
import History from "./pages/History";
import KeysManager from "./pages/KeysManager";
import Demo from "./pages/Demo";
import { IconLogo } from "./components/ui";

const navCls = ({ isActive }) =>
  `inline-flex min-h-[44px] items-center rounded-lg px-3 text-sm font-medium transition-colors duration-150 ${
    isActive
      ? "bg-white/15 text-white"
      : "text-brand-100 hover:bg-white/10 hover:text-white"
  }`;

export default function App() {
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-brand-700"
      >
        Skip to main content
      </a>

      <header className="bg-brand-900">
        <nav
          aria-label="Primary"
          className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-4 px-4 py-2 sm:px-6"
        >
          <NavLink
            to="/"
            end
            className="flex min-h-[44px] items-center gap-2 text-lg font-semibold tracking-tight text-white"
          >
            <IconLogo className="h-5 w-5 text-brand-300" />
            PaperOrchestra
          </NavLink>
          <div className="flex gap-1">
            <NavLink to="/" className={navCls} end>
              New Run
            </NavLink>
            <NavLink to="/history" className={navCls}>
              History
            </NavLink>
            <NavLink to="/demo" className={navCls}>
              Preview
            </NavLink>
            <NavLink to="/keys" className={navCls}>
              API Keys
            </NavLink>
          </div>
        </nav>
      </header>

      <main id="main" className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
        <Routes>
          <Route path="/" element={<NewRun />} />
          <Route path="/run/:id" element={<RunDetail />} />
          <Route path="/history" element={<History />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/keys" element={<KeysManager />} />
        </Routes>
      </main>
    </div>
  );
}
