export interface Student {
  id: string;
  name: string;
  seatNo?: number;
}

export interface DrawRecord {
  id: string;
  student: Student;
  timestamp: number;
}

export interface StudentGroup {
  id: string;
  name: string;
  colorIndex: number;
  leaderId?: string;
  members: Student[];
}

export type DrawMode = 'unique' | 'repeat'; // 不重複 | 可重複

export type GroupMode = 'bySize' | 'byCount'; // 每組幾人 | 分幾組
export type RemainderStrategy = 'distribute' | 'newGroup'; // 餘數分配給現有組 | 餘數獨立成組
