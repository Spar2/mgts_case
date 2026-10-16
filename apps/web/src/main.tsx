import { AppTheme, ThemeSwitch } from "./theme";
import { StrictMode, Suspense, lazy, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Loading } from "@carbon/react";
import { MotionConfig } from "motion/react";
import "./styles.scss";
import { ErrorBoundary } from "./error-boundary";
import { AccessGate, AccountActions } from "./access";
import { DemoGuardBanner } from "./demo-guard";
import { Activate, Landing } from "./public";
const Presentation = lazy(() => import("./presentation"));
const Explorer = lazy(() => import("./explorer"));
const Reviews = lazy(() => import("./reviews"));
const Candidates = lazy(() =>
  import("./workbench").then((m) => ({ default: m.Candidates })),
);
const Runs = lazy(() =>
  import("./workbench").then((m) => ({ default: m.Runs })),
);
const ControlPlane = lazy(() => import("./control-plane"));
const AccountSecurity = lazy(() => import("./account-security"));
const Cases = lazy(() => import("./cases"));
const Analysis = lazy(() => import("./analysis"));
const Pipeline = lazy(() => import("./pipeline"));
const Experiments = lazy(() => import("./experiments"));
const Investigations = lazy(() => import("./investigations"));
const Laboratory = lazy(() => import("./laboratory"));
const Monitoring = lazy(() => import("./monitoring"));
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30000, retry: 1, refetchOnWindowFocus: false },
  },
});
function Shell() {
  const [open, setOpen] = useState(false);
  return (
    <header className={`shell ${open ? "navigation-open" : ""}`}>
      <NavLink to="/" className="brand">
        Единая аналитическая платформа сенсорных данных
      </NavLink>
      <button
        className="mobile-menu"
        aria-expanded={open}
        aria-controls="main-navigation"
        onClick={() => setOpen((v) => !v)}
      >
        Меню
      </button>
      <nav
        id="main-navigation"
        aria-label="Основная навигация"
        onKeyDown={(e) => {if(e.key === 'Escape')setOpen(false);}}
        onClick={(e) => {if((e.target as HTMLElement).closest('a'))setOpen(false);}}
      >
        <NavLink to="/research">Исследование</NavLink>
        <NavLink to="/review">Экспертная проверка</NavLink>
        <NavLink to="/candidates">Кандидаты</NavLink>
        <NavLink to="/runs">Запуски</NavLink>
        <NavLink to="/analysis">ML-графики</NavLink>
        <NavLink to="/pipeline">Пайплайн</NavLink>
        <NavLink to="/cases">Кейсы</NavLink>
        <NavLink to="/demo">Демонстрация</NavLink>
        <NavLink to="/system">Система</NavLink>
        <NavLink to="/account">Аккаунт</NavLink>
      </nav>
      <span className="local-label">Командное рабочее место</span>
      <ThemeSwitch />
      <AccountActions />
    </header>
  );
}
function Application() {
  const path = useLocation().pathname;
  if (["/", "/activate", "/recover"].includes(path))
    return (
      <ErrorBoundary>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/activate" element={<Activate />} />
          <Route path="/recover" element={<Activate />} />
        </Routes>
      </ErrorBoundary>
    );
  return <WorkspaceApplication />;
}
function WorkspaceApplication() {
  return (
    <AccessGate>
      <Shell />
      <DemoGuardBanner />
      <ErrorBoundary>
        <Suspense
          fallback={
            <main className="page">
              <Loading
                description="Открываем рабочее место"
                withOverlay={false}
              />
            </main>
          }
        >
          <Routes>
            <Route path="/try" element={<Navigate to="/works" replace />} />
            <Route path="/request-access" element={<Navigate to="/works" replace />} />
            <Route path="/works" element={<Investigations />} />
            <Route path="/works/:id" element={<Investigations />} />
            <Route path="/laboratory" element={<Laboratory />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/research" element={<Explorer />} />
            <Route path="/review" element={<Reviews />} />
            <Route path="/candidates" element={<Candidates />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/pipeline" element={<Pipeline />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/cases" element={<Cases />} />
            <Route path="/cases/:id" element={<Cases />} />
            <Route path="/demo" element={<Presentation />} />
            <Route path="/system" element={<ControlPlane />} />
            <Route path="/account" element={<AccountSecurity />} />
            <Route path="*" element={<Navigate to="/research" replace />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
      <footer>
        Исходные наблюдения · происхождение сохранено · ответы окна не являются
        обучающими метками
      </footer>
    </AccessGate>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppTheme>
    <QueryClientProvider client={queryClient}>
      <MotionConfig reducedMotion="user">
        <BrowserRouter>
          <Application />
        </BrowserRouter>
      </MotionConfig>
    </QueryClientProvider>
    </AppTheme>
  </StrictMode>,
);
