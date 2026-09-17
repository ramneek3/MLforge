import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Dashboard" },
  { to: "/models", label: "Models" },
  { to: "/experiments", label: "Experiments" },
  { to: "/training", label: "Training" },
  { to: "/monitoring", label: "Monitoring" },
];

export function Sidebar() {
  return (
    <aside className="w-56 shrink-0 border-r border-line bg-panel min-h-screen p-5">
      <div className="mb-8">
        <h1 className="text-lg font-semibold tracking-wide">MLForge</h1>
        <p className="text-xs text-slate-400 mt-1">Churn MLOps platform</p>
      </div>
      <nav className="space-y-1">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `block rounded px-3 py-2 text-sm ${isActive ? "bg-accent/20 text-white" : "text-slate-300 hover:bg-white/5"}`
            }
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
