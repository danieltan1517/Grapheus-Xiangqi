main :: () {
  
  file, success := file_open("stuff.bin", for_writing=true, keep_existing_content=false, log_errors=true);
  if !success then {
    return;
  }

  defer {
    file_close(*file);
  }

  positions: [..] Position;
  array_reserve(*positions, 1_000_000);

  contents, success= := read_entire_file("xiangqi_evaluations.txt");
  _, _, contents = split_from_left(contents, #char "\n");

  i := 0;
  j := 0;
  while contents {
    found, line, contents= := split_from_left(contents, #char "\n");
    position := parse_line(line);
    array_add(*positions, position);
    i += 1;
    if i == 100_000 then {
      i = 0;
      j += 1;
      print("Position batch %, % positions created.\n", j, j * 100_000);
    }
  }

  header: DataSetHeader;
  header.entry_count = cast(u64) positions.count;
  file_write(*file, *header, size_of(DataSetHeader));
  file_write(*file, positions.data, cast(s64) (size_of(Position) * header.entry_count));
  
}

//181,9/2R1ak3/5a3/5n3/9/4R1P2/p7P/9/2Nr3c1/1r1AKCB2 b

PieceList :: struct {
    king: s8 = -1;
    advisor: [2] s8 = s8.[-1,-1];
    elephant: [2] s8 = s8.[-1,-1];
    knight: [2] s8 = s8.[-1,-1];
    rook: [2] s8 = s8.[-1,-1];
    cannon: [2] s8 = s8.[-1,-1];
    pawn: [5] s8 = s8.[-1,-1, -1, -1, -1];

};

add_piece :: (pieces: [] s8, sq: int) {
  for *p : pieces {
    if p.* == -1 {
      p.* = cast(s8) sq;
      return;
    }
  }
  assert(false, "Error. Too many pieces.");
}

Position :: struct {
    red:   PieceList;
    blk:   PieceList;
    turn:  s8;
    score: s16;
    wdl:   s8;
}

DataSetHeader :: struct {
    entry_count: u64;
    label_1: [128] u8;
    label_2: [128] u8;
    label_3: [1024] u8;
}

row_col_to_index :: inline (row: int, col: int) -> s8 {
  return cast(s8) (row + col * 9);
}

parse_line :: (line: string) -> Position {
    found, token, fen := split_from_left(line, #char ",");
    assert(found);
    position := fen_to_position(fen);
    score    := parse_int(*token);
    assert(score >= -900 && score <= 900);
    position.score = cast(s16) score;
    return position;
}

fen_to_position :: (fen: string) -> Position {
  position: Position;
  board_fen, success := parse_token(*fen);
  assert(success);

  col: u8 = 9;
  while true {
    found, token, rest := split_from_left(board_fen, #char "/");
    array:[] u8; 
    array.data = token.data;
    array.count = token.count;
    defer {
      board_fen = rest;
      col -= 1;
      if found == false then {
        break;
      }
    }

    row: u8 = 0;
    for letter: array {
      sq := row_col_to_index(row, col);
      if letter == {
      case #char "K";
        position.red.king = cast(s8) sq;
        row += 1;
      case #char "A";
        add_piece(position.red.advisor, sq);
        row += 1;
      case #char "B";
        add_piece(position.red.elephant, sq);
        row += 1;
      case #char "E";
        add_piece(position.red.elephant, sq);
        row += 1;
      case #char "R";
        add_piece(position.red.rook, sq);
        row += 1;
      case #char "C";
        add_piece(position.red.cannon, sq);
        row += 1;
      case #char "H";
        add_piece(position.red.knight, sq);
        row += 1;
      case #char "N";
        add_piece(position.red.knight, sq);
        row += 1;
      case #char "P";
        add_piece(position.red.pawn, sq);
        row += 1;

      case #char "k";
        position.blk.king = cast(s8) sq;
        row += 1;
      case #char "a";
        add_piece(position.blk.advisor, sq);
        row += 1;
      case #char "b";
        add_piece(position.blk.elephant, sq);
        row += 1;
      case #char "e";
        add_piece(position.blk.elephant, sq);
        row += 1;
      case #char "r";
        add_piece(position.blk.rook, sq);
        row += 1;
      case #char "c";
        add_piece(position.blk.cannon, sq);
        row += 1;
      case #char "h";
        add_piece(position.blk.knight, sq);
        row += 1;
      case #char "n";
        add_piece(position.blk.knight, sq);
        row += 1;
      case #char "p";
        add_piece(position.blk.pawn, sq);
        row += 1;
      case;
        assert(letter >= #char "1" && letter <= #char "9", "invalid letter % in fen %", to_string(*letter, 1), fen);
        letter -= #char "0";
        row += letter;
      }
    }

  }

  assert(position.red.king != -1, "There is no red king.");
  assert(position.blk.king != -1, "There is no red king.");

  // get the turn.
  turn_fen, success= := parse_token(*fen);
  assert(success);
  if equal(turn_fen, "w") {
    position.turn = 0;
  } else if equal(turn_fen, "b") {
    position.turn = 1;
  } else {
    assert(false);
  }

  return position;
}


#import "Basic";
#import "File";
#import "String";
