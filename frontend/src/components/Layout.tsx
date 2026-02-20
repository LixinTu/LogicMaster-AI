import { Outlet } from 'react-router-dom';
import { AppSidebar } from './AppSidebar';
import { TopBar } from './TopBar';
import { useAppStore } from '@/store/useAppStore';

export function Layout() {
  const intensity = useAppStore((s) => s.animationIntensity);
  return (
    <div className={`flex min-h-screen w-full scan-lines noise-bg sidebar-rainbow-bg${intensity === 'subtle' ? ' anim-subtle' : intensity === 'off' ? ' anim-off' : ''}`}>
      <AppSidebar />
      <div className="flex-1 flex flex-col min-h-screen relative z-10">
        <TopBar />
        <main className="flex-1 p-6 max-w-[1200px] mx-auto w-full static-rainbow-bg rounded-xl m-2">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
