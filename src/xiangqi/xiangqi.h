#pragma once
#include "../dataset/dataset.h"

namespace xiangqi {

struct Pieces {
    int8_t king;
    int8_t advisor[2];
    int8_t elephant[2];
    int8_t knight[2];
    int8_t rook[2];
    int8_t cannon[2];
    int8_t pawn[5];
};

struct Position : dataset::DataSetEntry {
    Pieces   red;
    Pieces   blk;
    int8_t   turn;
    int16_t  score;
    int8_t   wdl;
};

}
