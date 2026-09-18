import type {Metadata} from 'next';
import './globals.css'; // Global styles

export const metadata: Metadata = {
  title: '班級隨機抽籤與分組系統',
  description: '專為教師設計的教學輔助工具，支援 CSV 上傳與名單貼上，具備動畫與音效的隨機抽籤（可選重複/不重複）以及視覺化自動分組功能。',
  openGraph: {
    title: '班級隨機抽籤與分組系統',
    description: '專為教師設計的教學輔助工具，支援 CSV 上傳與名單貼上，具備動畫與音效的隨機抽籤（可選重複/不重複）以及視覺化自動分組功能。',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: '班級隨機抽籤與分組系統',
    description: '專為教師設計的教學輔助工具，支援 CSV 上傳與名單貼上，具備動畫與音效的隨機抽籤（可選重複/不重複）以及視覺化自動分組功能。',
  },
};

export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="zh-Hant">
      <body suppressHydrationWarning className="bg-slate-50 text-slate-900 antialiased min-h-screen">{children}</body>
    </html>
  );
}
