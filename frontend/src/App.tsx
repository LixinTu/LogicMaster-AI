import { useState, useEffect } from 'react';
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Practice from "./pages/Practice";
import WrongBook from "./pages/WrongBook";
import Analytics from "./pages/Analytics";
import SettingsPage from "./pages/Settings";
import ProfilePage from "./pages/Profile";
import LoginPage from "./pages/Login";
import NotFound from "./pages/NotFound";
import { IntroAnimation } from "./components/IntroAnimation";
import { useAuthStore } from "./store/useAuthStore";

const queryClient = new QueryClient();

function AuthenticatedApp() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/practice" element={<Practice />} />
        <Route path="/wrong-book" element={<WrongBook />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/profile" element={<ProfilePage />} />
      </Route>
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}

function AppContent() {
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore();
  const [showIntro, setShowIntro] = useState(() => !sessionStorage.getItem('intro_played'));

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  if (showIntro) {
    return <IntroAnimation onComplete={() => setShowIntro(false)} />;
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'hsl(270,100%,5%)' }}>
        <div className="loading-bulb-container">
          <div className="loading-bulb-icon rainbow-spinner w-8 h-8" />
        </div>
      </div>
    );
  }

  return isAuthenticated ? <AuthenticatedApp /> : <Routes><Route path="*" element={<LoginPage />} /></Routes>;
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
