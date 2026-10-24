import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AuthContext, useAuth } from "./lib/useAuth.jsx";
import { currentSession } from "./lib/auth.js";
import Landing from "./pages/Landing.jsx";
import Login from "./pages/Login.jsx";
import Evidence from "./pages/Evidence.jsx";
import About from "./pages/About.jsx";
import PublicShell from "./layout/PublicShell.jsx";
import PatientShell from "./layout/PatientShell.jsx";
import ClinicianShell from "./layout/ClinicianShell.jsx";
import PatientAssess from "./pages/patient/Assess.jsx";
import PatientHistory from "./pages/patient/History.jsx";
import Learn from "./pages/patient/Learn.jsx";
import ClinicianAssess from "./pages/clinician/Assess.jsx";
import Records from "./pages/clinician/Records.jsx";
import Batch from "./pages/clinician/Batch.jsx";

function Guard({ role, children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to={`/login/${role}`} replace />;
  if (user.role !== role) return <Navigate to={`/${user.role}`} replace />;
  return children;
}

function Home() {
  const { user } = useAuth();
  return user ? <Navigate to={`/${user.role}`} replace /> : <Landing />;
}

export default function App() {
  const [user, setUser] = useState(currentSession);
  return (
    <AuthContext.Provider value={{ user, setUser }}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login/:role" element={<Login />} />
        <Route element={<PublicShell />}>
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/about" element={<About />} />
        </Route>
        <Route path="/patient" element={<Guard role="patient"><PatientShell /></Guard>}>
          <Route index element={<PatientAssess />} />
          <Route path="history" element={<PatientHistory />} />
          <Route path="learn" element={<Learn />} />
          <Route path="evidence" element={<Evidence />} />
        </Route>
        <Route path="/clinician" element={<Guard role="clinician"><ClinicianShell /></Guard>}>
          <Route index element={<ClinicianAssess />} />
          <Route path="batch" element={<Batch />} />
          <Route path="records" element={<Records />} />
          <Route path="evidence" element={<Evidence />} />
          <Route path="method" element={<About />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthContext.Provider>
  );
}
