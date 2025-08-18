import { describe, expect, it } from 'vitest';
import { validMoves } from './index';

describe('validMoves', () => {
  it('returns empty array for placeholder', () => {
    expect(validMoves({ board: [] })).toEqual([]);
  });
});
