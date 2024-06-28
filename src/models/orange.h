#pragma once

#include "../xiangqi/xiangqi.h"
#include "chessmodel.h"

namespace model {

static int mirror_vert[90] = {
  81, 82, 83, 84, 85, 86, 87, 88, 89,
  72, 73, 74, 75, 76, 77, 78, 79, 80,
  63, 64, 65, 66, 67, 68, 69, 70, 71,
  54, 55, 56, 57, 58, 59, 60, 61, 62,
  45, 46, 47, 48, 49, 50, 51, 52, 53, 
  36, 37, 38, 39, 40, 41, 42, 43, 44, 
  27, 28, 29, 30, 31, 32, 33, 34, 35,
  18, 19, 20, 21, 22, 23, 24, 25, 26,
   9, 10, 11, 12, 13, 14, 15, 16, 17,
   0,  1,  2,  3,  4,  5,  6,  7,  8
};

static int king_sq_index[90] = {
   -1, -1, -1,  1,  0,  1, -1, -1, -1,
   -1, -1, -1,  1,  1,  1, -1, -1, -1,
   -1, -1, -1,  1,  1,  1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1,  1,  1,  1, -1, -1, -1,
   -1, -1, -1,  1,  1,  1, -1, -1, -1,
   -1, -1, -1,  1,  0,  1, -1, -1, -1,
};

// stm = side to move. 0=red. 1=black.
static int get_piece_type(int p, int stm) {
    if (p == 0) {
        return 0;
    }
    if (stm == 0) {
        return p;
    }
    if (p >= 5) {
        return p - 4;
    } else {
        return p + 4;
    }
}

int index_values(int stm, int ksq, int piece, int sq) {
    piece = get_piece_type(piece, stm);
    return (ksq * 9 * 90) + (piece * 90) + sq;
}

struct OrangeModel : Model {

    float lambda;
    SparseInput* in1;
    SparseInput* in2;

    OrangeModel(float lambda, size_t save_rate) {
	this->lambda = lambda;
        in1     = add<SparseInput>(2 * 9 * 90, 32);
        in2     = add<SparseInput>(2 * 9 * 90, 32);

        auto ft = add<FeatureTransformer>(in1, in2, 128);
        auto re = add<ReLU>(ft);
        auto af = add<Affine>(re, 1);
        auto sm = add<Sigmoid>(af, 1.0 / 360.0);
        add_optimizer(Adam({{OptimizerEntry {&ft->weights}},
                            {OptimizerEntry {&ft->bias}},
                            {OptimizerEntry {&af->weights}},
                            {OptimizerEntry {&af->bias}}},
                           0.9,
                           0.999,
                           1e-8));

        set_save_frequency(save_rate);

	// similar quantization scheme to Stockfish.
        add_quantization(Quantizer {
            "quant_1",
            10,
            QuantizerEntry<int16_t>(&ft->weights.values, 32, true),
            QuantizerEntry<int16_t>(&ft->bias.values, 32),
            QuantizerEntry<int16_t>(&af->weights.values, 128),
            QuantizerEntry<int32_t>(&af->bias.values, 128 * 32),
        });
    }

    using BatchLoader = dataset::BatchLoader<xiangqi::Position>;

    /**
     * @brief Trains the model using the provided train and validation loaders for a specified number
     * of epochs.
     *
     * @param train_loader The batch loader for training data.
     * @param val_loader The batch loader for validation data (optional).
     * @param epochs Number of training epochs (default: 1500).
     * @param epoch_size Number of batches per epoch (default: 1e8).
     */
    void train(BatchLoader&                train_loader,
               std::optional<BatchLoader>& val_loader,
               int                         epochs         = 1500,
               int                         epoch_size     = 1e8,
               int                         val_epoch_size = 1e7) {

        this->compile(train_loader.batch_size);

        Timer t {};
        for (int i = 1; i <= epochs; i++) {
            t.start();

            uint64_t prev_print_tm    = 0;
            float    total_epoch_loss = 0;
            float    total_val_loss   = 0;

            // Training phase
            for (int b = 1; b <= epoch_size / train_loader.batch_size; b++) {
                auto* ds = train_loader.next();
                setup_inputs_and_outputs(ds);

                float batch_loss = batch();
                total_epoch_loss += batch_loss;
                float epoch_loss = total_epoch_loss / b;

                t.stop();
                uint64_t elapsed = t.elapsed();
                if (elapsed - prev_print_tm > 1000 || b == epoch_size / train_loader.batch_size) {
                    prev_print_tm = elapsed;

                    printf("\rep = [%4d], epoch_loss = [%1.8f], batch = [%5d], batch_loss = [%1.8f], "
                           "speed = [%7.2f it/s], time = [%3ds]",
                           i,
                           epoch_loss,
                           b,
                           batch_loss,
                           1000.0f * b / elapsed,
                           (int) (elapsed / 1000.0f));
                    std::cout << std::flush;
                }
            }

            // Validation phase (if validation loader is provided)
            if (val_loader.has_value()) {
                for (int b = 1; b <= val_epoch_size / val_loader->batch_size; b++) {
                    auto* ds = val_loader->next();
                    setup_inputs_and_outputs(ds);

                    float val_batch_loss = loss();
                    total_val_loss += val_batch_loss;
                }
            }

            float epoch_loss = total_epoch_loss / (val_epoch_size / train_loader.batch_size);
            float val_loss   = (val_loader.has_value())
                                   ? total_val_loss / (val_epoch_size / val_loader->batch_size)
                                   : 0;

            printf(", val_loss = [%1.8f] ", val_loss);
            next_epoch(epoch_loss, val_loss);
            std::cout << std::endl;
        }
    }

    void setup_piece(int b, int turn, int red_king, int blk_king, int8_t pieces_squares[], int num, int piece_type) {
        const int RED_TURN = 0;
	for(int i=0; i<num; i++) {
            int sq = pieces_squares[i];
            if (sq == -1) {
                continue;
	    }
            int piece_index_white_pov = index_values(0, red_king, piece_type, sq);
            int piece_index_black_pov = index_values(1, blk_king, piece_type, mirror_vert[sq]);
            if (turn == RED_TURN) {
                in1->sparse_output.set(b, piece_index_white_pov);
                in2->sparse_output.set(b, piece_index_black_pov);
            } else {
                in2->sparse_output.set(b, piece_index_white_pov);
                in1->sparse_output.set(b, piece_index_black_pov);
            }
        }
    }

    void setup_inputs_and_outputs(dataset::DataSet<xiangqi::Position>* positions) {
        //const int BLK_TURN = 1;
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto &target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(4)
        for (int b = 0; b < positions->header.entry_count; b++) {
            xiangqi::Position* pos = &positions->positions[b];
            // fill in the inputs and target values
	    int8_t red_king = king_sq_index[pos->red.king];
	    int8_t blk_king = king_sq_index[pos->blk.king];
            int turn = pos->turn;

	    const int ADVISOR_ELE_OP = 0;
	    // advisors
	    setup_piece(b, turn, red_king, blk_king, pos->red.advisor,  2, ADVISOR_ELE_OP);
	    setup_piece(b, turn, red_king, blk_king, pos->blk.advisor,  2, ADVISOR_ELE_OP);

	    // elephants
	    setup_piece(b, turn, red_king, blk_king, pos->red.elephant, 2, ADVISOR_ELE_OP);
	    setup_piece(b, turn, red_king, blk_king, pos->blk.elephant, 2, ADVISOR_ELE_OP);

	    const int R_KNIGHT_OP = 1;
	    const int B_KNIGHT_OP = 5;

	    // knight
	    setup_piece(b, turn, red_king, blk_king, pos->red.knight, 2, R_KNIGHT_OP);
	    setup_piece(b, turn, red_king, blk_king, pos->blk.knight, 2, B_KNIGHT_OP);

	    const int R_ROOK = 2;
	    const int B_ROOK = 6;
	    // rook
	    setup_piece(b, turn, red_king, blk_king, pos->red.rook, 2, R_ROOK);
	    setup_piece(b, turn, red_king, blk_king, pos->blk.rook, 2, B_ROOK);
          
	    const int R_CANNON = 3;
	    const int B_CANNON = 7;
	    // cannon 
	    setup_piece(b, turn, red_king, blk_king, pos->red.cannon, 2, R_CANNON);
	    setup_piece(b, turn, red_king, blk_king, pos->blk.cannon, 2, B_CANNON);

	    const int R_PAWN = 4;
	    const int B_PAWN = 8;
	    // pawn 
	    setup_piece(b, turn, red_king, blk_king, pos->red.pawn, 5, R_PAWN);
	    setup_piece(b, turn, red_king, blk_king, pos->blk.pawn, 5, B_PAWN);

	    // the data files we load in are already doing evaluation relative to the side.
            float p_value = (float)(pos->score) / 360.0;
	    //float w_value = pos->wdl;
	    //
	    // sigmoid.
	    float p_target = 1.0 / (1.0 + expf(-p_value));
	    target(b) = p_target;
        }
    }
};

}    // namespace model
