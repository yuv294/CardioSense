import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AuthContext, useAuth } from "./lib/useAuth.jsx";
import { currentSession } from "./lib/auth.js";
import Gate from "./pages/Gate.jsx";
import Login from "./pages/Login.jsx";
import PatientShell from "./layout/PatientShell.jsx";
import ClinicianShell from "./layout/ClinicianShell.jsx";
import PatientAssess from "./pages/patient/Assess.jsx";
import PatientHistory from "./pages/patient/History.jsx";
import Learn from "./pages/patient/Learn.jsx";
import ClinicianAssess from "./pages/clinician/Assess.jsx";
import Records from "./pages/clinician/Records.jsx";
import Performance from "./pages/clinician/Performance.jsx";

function Guard({ role, children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to={`/login/${role}`} replace />;
  if (user.role !== role) return <Navigate to={`/${user.role}`} replace />;
  return children;
}

function Home() {
  const { user } = useAuth();
  return user ? <Navigate to={`/${user.role}`} replace /> : <Gate />;
}

export default function App() {
  const [user, setUser] = useState(currentSession);
  return (
    <AuthContext.Provider value={{ user, setUser }}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login/:role" element={<Login />} />
        <Route path="/patient" element={<Guard role="patient"><PatientShell /></Guard>}>
          <Route index element={<PatientAssess />} />
          <Route path="history" element={<PatientHistory />} />
          <Route path="learn" element={<Learn />} />
        </Route>
        <Route path="/clinician" element={<Guard role="clinician"><ClinicianShell /></Guard>}>
          <Route index element={<ClinicianAssess />} />
          <Route path="records" element={<Records />} />
          <Route path="performance" element={<Performance />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthContext.Provider>
  );
}
