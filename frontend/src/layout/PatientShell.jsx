import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../lib/useAuth.jsx";
import { endSession } from "../lib/auth.js";
import { Wordmark } from "../components/Ecg.jsx";

const NAV = [
  { to: "/patient", label: "Assess", end: true },
  { to: "/patient/history", label: "History" },
  { to: "/patient/learn", label: "Learn" },
  { to: "/patient/evidence", label: "Evidence" },
];

export default function PatientShell() {
  const { user, setUser } = useAuth();
  return (
    <div data-role="patient" className="min-h-screen">
      <header className="border-b border-fg/10 bg-surface">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <Wordmark className="text-lg" />
          <nav className="flex items-center gap-1 overflow-x-auto">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.end}
                className={({ isActive }) =>
                  `rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                    isActive ? "bg-fg text-surface" : "text-fg/65 hover:bg-fg/5 hover:text-fg"}`}>
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-4 text-sm">
            <span className="hidden text-fg/70 sm:inline">{user.name}</span>
            <button onClick={() => { endSession(); setUser(null); }}
                    className="font-medium text-accent hover:underline">Sign out</button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-10"><Outlet /></main>
    </div>
  );
}
