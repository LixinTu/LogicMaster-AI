import { useLocation } from 'react-router-dom';
import { useAppStore } from '@/store/useAppStore';
import { AnimatedNumber } from './AnimatedNumber';
import { Flame } from 'lucide-react';

const pageTitles: Record<string, string> = {
  '/': 'Practice',
  '/practice': 'Practice',
  '/analytics': 'Analytics',
  '/review': 'Review',
  '/settings': 'Settings',
};

export function TopBar() {
  const location = useLocation();
  const { theta, gmatScore, streak } = useAppStore();
  const title = pageTitles[location.pathname] || 'Practice';

  return (
    <header className="h-14 flex items-center justify-between px-6"
      style={{ background: 'linear-gradient(90deg, hsl(270, 100%, 8%), hsl(240, 80%, 6%), hsl(220, 100%, 5%))' }}>
      <div className="flex items-center">
        <span className="topbar-logo-glitch">
          LogicMaster AI
        </span>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 rounded-full px-4 py-1.5"
          style={{ background: 'hsl(56, 100%, 50%)' }}>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono uppercase" style={{ color: 'hsl(270, 100%, 8%, 0.6)' }}>Theta</span>
            <span className="text-sm font-bold font-mono" style={{ color: 'hsl(270, 100%, 8%)' }}>
              <AnimatedNumber value={theta} decimals={2} />
            </span>
          </div>
          <div className="w-px h-4" style={{ background: 'hsl(270, 100%, 8%, 0.2)' }} />
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono uppercase" style={{ color: 'hsl(270, 100%, 8%, 0.6)' }}>GMAT</span>
            <span className="text-sm font-bold font-mono" style={{ color: 'hsl(270, 100%, 8%)' }}>
              <AnimatedNumber value={gmatScore} />
            </span>
          </div>
          <div className="w-px h-4" style={{ background: 'hsl(270, 100%, 8%, 0.2)' }} />
          <div className="flex items-center gap-1.5">
            <Flame size={14} style={{ color: 'hsl(270, 100%, 8%)' }} />
            <span className="text-sm font-bold font-mono" style={{ color: 'hsl(270, 100%, 8%)' }}>
              <AnimatedNumber value={streak} />
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
