import { motion } from 'framer-motion';
import { useAppStore } from '@/store/useAppStore';

interface GlitchTextProps {
  children: string;
  className?: string;
  as?: 'h1' | 'h2' | 'h3' | 'h4' | 'span' | 'p';
}

export function GlitchText({ children, className = '', as: Tag = 'span' }: GlitchTextProps) {
  const intensity = useAppStore((s) => s.animationIntensity);

  if (intensity === 'off') {
    return <Tag className={className}>{children}</Tag>;
  }

  return (
    <motion.span
      className={`inline-block ${className}`}
      whileHover={intensity === 'full' ? {
        textShadow: [
          'none',
          '2px 0 hsl(345, 100%, 60%), -2px 0 hsl(180, 100%, 50%)',
          '-1px 0 hsl(345, 100%, 60%), 1px 0 hsl(180, 100%, 50%)',
          'none',
        ],
        transition: { duration: 0.15, repeat: 1 },
      } : undefined}
    >
      <Tag className={className}>{children}</Tag>
    </motion.span>
  );
}
