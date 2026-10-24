import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../lib/useAuth.jsx";
import { endSession } from "../lib/auth.js";
import { Wordmark } from "../components/Ecg.jsx";

const NAV = [
  { to: "/clinician", label: "Assess patient", end: true },
  { to: "/clinician/batch", label: "Batch scoring" },
  { to: "/clinician/records", label: "Patient records" },
  { to: "/clinician/evidence", label: "Evidence" },
  { to: "/clinician/method", label: "Method" },
];

export default function ClinicianShell() {
  const { user, setUser } = useAuth();
  return (
    <div data-role="clinician" className="min-h-screen md:flex">
      <aside className="border-b border-fg/10 bg-surface md:sticky md:top-0 md:flex md:h-screen md:w-64 md:shrink-0 md:flex-col md:border-b-0 md:border-r">
        <div className="px-6 py-5"><Wordmark className="text-lg" /></div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-1 md:flex-col md:pb-0">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive ? "bg-accent/15 text-accent" : "text-fg/70 hover:bg-fg/5 hover:text-fg"}`}>
              {n.label}
            </NavLink>
          ))}
          <button onClick={() => { endSession(); setUser(null); }}
                  className="whitespace-nowrap rounded-lg px-3 py-2.5 text-sm font-medium text-accent md:hidden">Sign out</button>
        </nav>
        <div className="hidden border-t border-fg/10 p-5 text-sm md:block">
          <div className="font-medium">{user.name}</div>
          <div className="truncate text-fg/50">{user.email}</div>
          <button onClick={() => { endSession(); setUser(null); }}
                  className="mt-3 font-medium text-accent hover:underline">Sign out</button>
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-6 py-8 md:px-10"><Outlet /></main>
    </div>
  );
}
