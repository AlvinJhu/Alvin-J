import { Student } from './types';

export const DEFAULT_STUDENTS: Student[] = [
  { id: 'std-1', seatNo: 1, name: '陳柏翰' },
  { id: 'std-2', seatNo: 2, name: '林怡君' },
  { id: 'std-3', seatNo: 3, name: '張家豪' },
  { id: 'std-4', seatNo: 4, name: '黃雅婷' },
  { id: 'std-5', seatNo: 5, name: '李冠宇' },
  { id: 'std-6', seatNo: 6, name: '王品涵' },
  { id: 'std-7', seatNo: 7, name: '吳承恩' },
  { id: 'std-8', seatNo: 8, name: '蔡欣妤' },
  { id: 'std-9', seatNo: 9, name: '劉宇軒' },
  { id: 'std-10', seatNo: 10, name: '楊佩蓉' },
  { id: 'std-11', seatNo: 11, name: '許哲銘' },
  { id: 'std-12', seatNo: 12, name: '鄭婷婷' },
  { id: 'std-13', seatNo: 13, name: '謝宗翰' },
  { id: 'std-14', seatNo: 14, name: '郭靜雯' },
  { id: 'std-15', seatNo: 15, name: '洪偉哲' },
  { id: 'std-16', seatNo: 16, name: '曾詩涵' },
  { id: 'std-17', seatNo: 17, name: '邱建良' },
  { id: 'std-18', seatNo: 18, name: '廖依婷' },
  { id: 'std-19', seatNo: 19, name: '賴柏宇' },
  { id: 'std-20', seatNo: 20, name: '周羽萱' },
  { id: 'std-21', seatNo: 21, name: '徐志豪' },
  { id: 'std-22', seatNo: 22, name: '蘇若晴' },
  { id: 'std-23', seatNo: 23, name: '葉俊廷' },
  { id: 'std-24', seatNo: 24, name: '莊宛蓉' },
  { id: 'std-25', seatNo: 25, name: '江彥廷' },
  { id: 'std-26', seatNo: 26, name: '呂亭宜' },
  { id: 'std-27', seatNo: 27, name: '何睿哲' },
  { id: 'std-28', seatNo: 28, name: '羅晨曦' },
  { id: 'std-29', seatNo: 29, name: '高詠晴' },
  { id: 'std-30', seatNo: 30, name: '潘宥廷' }
];

/**
 * Parses raw text input (from paste area or CSV file text).
 * Automatically detects CSV formats (with seat number/name columns) or plain newline/comma lists.
 */
export function parseRosterText(rawText: string): Student[] {
  if (!rawText || !rawText.trim()) return [];

  const lines = rawText
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(l => l.length > 0);

  const students: Student[] = [];
  let currentSeat = 1;

  // Check if first line looks like a header (e.g. contains "姓名", "name", "座號", "學號", "student")
  let startIndex = 0;
  if (lines.length > 0) {
    const firstLineLower = lines[0].toLowerCase();
    if (
      firstLineLower.includes('姓名') ||
      firstLineLower.includes('name') ||
      firstLineLower.includes('座號') ||
      firstLineLower.includes('學號') ||
      firstLineLower.includes('student')
    ) {
      startIndex = 1;
    }
  }

  for (let i = startIndex; i < lines.length; i++) {
    const line = lines[i];

    // Check if comma or tab separated
    let parts: string[] = [];
    if (line.includes(',')) {
      parts = line.split(',').map(p => p.trim().replace(/^["']|["']$/g, ''));
    } else if (line.includes('\t')) {
      parts = line.split('\t').map(p => p.trim());
    } else if (line.includes('，')) {
      parts = line.split('，').map(p => p.trim());
    } else {
      // Could be whitespace separated or single item
      parts = [line];
    }

    // Process parts
    if (parts.length >= 2) {
      // Likely SeatNumber, Name or ID, Name
      let seatCandidate: number | undefined;
      let nameCandidate = '';

      // Check if first part is a number
      const p0Num = parseInt(parts[0], 10);
      if (!isNaN(p0Num) && p0Num > 0 && p0Num < 1000) {
        seatCandidate = p0Num;
        nameCandidate = parts[1];
      } else {
        // Maybe name is first, or check other parts
        nameCandidate = parts.find(p => p.length > 0 && isNaN(Number(p))) || parts[0];
      }

      if (nameCandidate && nameCandidate.length > 0) {
        students.push({
          id: `std-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          seatNo: seatCandidate || currentSeat++,
          name: nameCandidate
        });
        continue;
      }
    }

    // Otherwise each item on the line could be a student name, or comma-separated names on same line
    for (const part of parts) {
      // Remove leading serial numbers like "1. 王小明" or "01 王小明" or "(1) 王小明"
      const cleaned = part.replace(/^[\d\s.\-、#（）()\[\]]+\s*/, '').trim();
      const extractedSeat = part.match(/^(\d+)[\s.\-、#]/)?.[1];
      const parsedSeat = extractedSeat ? parseInt(extractedSeat, 10) : undefined;

      if (cleaned.length > 0) {
        students.push({
          id: `std-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          seatNo: parsedSeat || currentSeat++,
          name: cleaned
        });
      }
    }
  }

  // If no seat numbers were assigned, number them consecutively
  return students.map((s, idx) => ({
    ...s,
    seatNo: s.seatNo || idx + 1
  }));
}
