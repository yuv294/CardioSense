import { Link, Outlet } from "react-router-dom";
import { Wordmark } from "../components/Ecg.jsx";

export default function PublicShell() {
  return (
    <div data-role="patient" className="min-h-screen">
      <header className="border-b border-fg/10 bg-surface">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <Link to="/"><Wordmark className="text-lg" /></Link>
          <nav className="flex gap-5 text-sm font-medium text-fg/65">
            <Link to="/evidence" className="hover:text-fg">Evidence</Link>
            <Link to="/about" className="hover:text-fg">Method</Link>
            <Link to="/login/patient" className="text-accent hover:underline">Check my risk</Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-10"><Outlet /></main>
    </div>
  );
}
