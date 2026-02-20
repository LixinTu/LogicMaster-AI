import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Home, Zap, Bug, BarChart3, Settings, ChevronLeft, ChevronRight, Lightbulb, LogOut, UserCircle } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { api } from '@/lib/api';

const mainNavItems = [
  { title: 'Dashboard', path: '/dashboard', icon: Home },
  { title: 'Practice', path: '/practice', icon: Zap },
  { title: 'Wrong Book', path: '/wrong-book', icon: Bug },
];

const secondaryNavItems = [
  { title: 'Analytics', path: '/analytics', icon: BarChart3 },
  { title: 'Profile', path: '/profile', icon: UserCircle },
  { title: 'Settings', path: '/settings', icon: Settings },
];

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const [health, setHealth] = useState<{ api: boolean; db: boolean; qdrant: boolean }>({
    api: false, db: false, qdrant: false,
  });
  const location = useLocation();
  const { user, logout } = useAuthStore();

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await api.health();
        setHealth({
          api: res.status === 'ok',
          db: res.db_status === 'connected',
          qdrant: res.qdrant_status === 'connected',
        });
      } catch {
        setHealth({ api: false, db: false, qdrant: false });
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const renderNavItem = (item: typeof mainNavItems[0], isSecondary = false) => {
    const isActive = location.pathname === item.path || (item.path === '/dashboard' && location.pathname === '/');
    return (
      <NavLink
        key={item.path}
        to={item.path}
        className={`flex items-center gap-3 px-3 py-2.5 rounded-md relative group transition-all duration-150 ${
          isActive
            ? 'sidebar-rainbow-bg text-primary-foreground font-bold shadow-[0_0_24px_hsl(56,100%,50%,0.5)] chromatic-text'
            : isSecondary
            ? 'text-primary/60 hover:text-primary hover:bg-primary/10 hover:shadow-[0_0_20px_hsl(56,100%,50%,0.3)] active:scale-95 hover:scale-[1.02]'
            : 'text-primary/80 hover:text-primary hover:bg-primary/10 hover:shadow-[0_0_30px_hsl(56,100%,50%,0.6),0_0_60px_hsl(56,100%,50%,0.2)] active:scale-95 hover:scale-[1.04]'
        }`}
      >
        <item.icon size={isSecondary ? 18 : 20} className="flex-shrink-0" />
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className={`font-mono tracking-wide ${isSecondary ? 'text-xs' : 'text-sm'}`}
            >
              {item.title}
            </motion.span>
          )}
        </AnimatePresence>
      </NavLink>
    );
  };

  return (
    <motion.aside
      className="h-screen border-r border-border flex flex-col sticky top-0 z-40 overflow-hidden"
      style={{ background: 'linear-gradient(180deg, hsl(270, 100%, 8%) 0%, hsl(240, 80%, 6%) 50%, hsl(220, 100%, 5%) 100%)' }}
      animate={{ width: collapsed ? 60 : 260 }}
      transition={{ duration: 0.2 }}
    >
      {/* Logo */}
      <div className="p-4 flex items-center gap-3 border-b border-border">
        <motion.div
          className="text-primary flex-shrink-0"
          animate={{ filter: 'brightness(0.7)' }}
          transition={{ duration: 1.5, repeat: Infinity, repeatType: 'reverse' }}
        >
          <Lightbulb size={28} />
        </motion.div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <div className="font-heading text-lg font-bold tracking-wider rainbow-logo-text uppercase">GlitchMind</div>
              <div className="font-mono text-xs text-foreground/70">
                {user?.display_name || 'Demo User'}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Main nav */}
      <nav className="flex-1 p-2 space-y-1 mt-2">
        {mainNavItems.map((item) => renderNavItem(item))}

        {/* Separator */}
        <div className="my-3 mx-2 border-t border-border/40" />

        {secondaryNavItems.map((item) => renderNavItem(item, true))}
      </nav>

      {/* System health */}
      <div className="p-3 border-t border-border" style={{ backgroundColor: 'hsl(0, 0%, 0%, 0.3)' }}>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-2"
            >
              <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">System</div>
              {[
                { label: 'API', ok: health.api },
                { label: 'DB', ok: health.db },
                { label: 'Qdrant', ok: health.qdrant },
              ].map((s) => (
                <div key={s.label} className="flex items-center gap-2.5 text-xs font-mono text-muted-foreground">
                  <div className={`health-dot ${s.ok ? 'health-dot-ok' : 'health-dot-err'}`} />
                  {s.label}
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
        {collapsed && (
          <div className="flex flex-col items-center gap-2">
            {[health.api, health.db, health.qdrant].map((ok, i) => (
              <div key={i} className={`health-dot ${ok ? 'health-dot-ok' : 'health-dot-err'}`} />
            ))}
          </div>
        )}
      </div>

      {/* Logout */}
      <button
        onClick={logout}
        className="flex items-center gap-3 px-3 py-2.5 mx-2 mb-1 rounded-md text-destructive/70 hover:text-destructive hover:bg-destructive/10 transition-all duration-150"
      >
        <LogOut size={20} className="flex-shrink-0" />
        <AnimatePresence>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="font-mono text-sm tracking-wide"
            >
              LOGOUT
            </motion.span>
          )}
        </AnimatePresence>
      </button>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="p-2 border-t border-border text-muted-foreground hover:text-foreground transition-colors flex items-center justify-center"
      >
        {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>
    </motion.aside>
  );
}
