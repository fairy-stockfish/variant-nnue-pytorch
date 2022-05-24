/*

Copyright 2020 Tomasz Sobczyk

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#pragma once

#include <bitset>

#include <cstdio>
#include <cassert>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <iostream>
#include <set>
#include <cstdio>
#include <cassert>
#include <array>
#include <limits>
#include <climits>
#include <optional>
#include <thread>
#include <mutex>
#include <random>

#include "rng.h"

#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && !defined(__clang__)
#include <intrin.h>
#endif

#define FILES 9
#define RANKS 10
#define PIECE_TYPES 7
#define PIECE_COUNT 32
#define POCKETS false
#define KING_SQUARES 9

namespace chess
{
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)

    #define FORCEINLINE __attribute__((always_inline))

    #elif defined(_MSC_VER)

    // NOTE: for some reason it breaks the profiler a little
    //       keep it on only when not profiling.
    //#define FORCEINLINE __forceinline
    #define FORCEINLINE

    #else

    #define FORCEINLINE inline

    #endif

    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)

    #define NOINLINE __attribute__((noinline))

    #elif defined(_MSC_VER)

    #define NOINLINE __declspec(noinline)

    #else

    #define NOINLINE

    #endif

    enum struct Color : std::uint8_t
    {
        White,
        Black,
        NB
    };

    enum struct PieceType : std::uint8_t
    {
        Pawn,
        Knight,
        Bishop,
        Rook,
        Queen,
        MaxPiece = PIECE_TYPES - 1,
        King = MaxPiece + (KING_SQUARES == 1),

        None,
        NB = King + 1
    };

    enum struct Piece : std::uint8_t
    {
        White = 0,
        Black = static_cast<uint8_t>(PieceType::King) + 1,
        Max = Black + static_cast<uint8_t>(PieceType::King),
        None,
        NB
    };

    constexpr Color color_of(Piece p) {
        return p >= Piece::Black ? Color::Black : Color::White;
    }

    constexpr PieceType type_of(Piece p) {
        return static_cast<PieceType>(p >= Piece::Black ? static_cast<uint8_t>(p) - static_cast<uint8_t>(Piece::Black) : static_cast<uint8_t>(p));
    }

    constexpr Piece make_piece(PieceType pt, Color c) {
        return static_cast<Piece>(static_cast<uint8_t>(c == Color::Black) * static_cast<uint8_t>(Piece::Black) + static_cast<uint8_t>(pt));
    }

    enum struct Rank : std::int8_t // make sure that it does not underflow in loop
    {
        RANK_1,
        RANK_2,
        RANK_3,
        RANK_4,
        RANK_5,
        RANK_6,
        RANK_7,
        RANK_8,
        RANK_MAX = RANKS - 1,
        RANK_NB,
    };

    enum struct File : std::uint8_t
    {
        FILE_A,
        FILE_B,
        FILE_C,
        FILE_D,
        FILE_E,
        FILE_F,
        FILE_G,
        FILE_H,
        FILE_MAX = FILES - 1,
        FILE_NB
    };

    enum struct Square : std::uint8_t
    {
        MIN = 0,
        NB = std::uint8_t(Rank::RANK_NB) * std::uint8_t(File::FILE_NB),
        MAX = NB - 1,
        KNB = KING_SQUARES,
    };

    inline Square make_square(File f, Rank r) {
        return Square(std::uint8_t(File::FILE_NB) * std::uint8_t(r) + std::uint8_t(f));
    }

    inline Rank file_of(Square s) {
        return Rank(std::uint8_t(s) % std::uint8_t(File::FILE_NB));
    }

    inline Rank rank_of(Square s) {
        return Rank(std::uint8_t(s) / std::uint8_t(File::FILE_NB));
    }

    inline Square flip_horizontally(Square s) {
        return Square(std::uint8_t(s) + std::uint8_t(File::FILE_MAX) - 2 * std::uint8_t(file_of(s)));
    }

    inline Square flip_vertically(Square s) {
        return Square(std::int8_t(s) + (std::int8_t(Rank::RANK_MAX) - 2 * std::int8_t(rank_of(s))) * std::int8_t(File::FILE_NB));
    }

    #define ENABLE_INCR_OPERATORS_ON(T)                                \
    inline T& operator++(T& d) { return d = T(int(d) + 1); }           \
    inline T& operator--(T& d) { return d = T(int(d) - 1); }
    ENABLE_INCR_OPERATORS_ON(Square);
    ENABLE_INCR_OPERATORS_ON(File);
    ENABLE_INCR_OPERATORS_ON(Rank);
    ENABLE_INCR_OPERATORS_ON(PieceType);

    enum struct MoveType : std::uint8_t
    {
        Normal,
        Promotion,
        Castle,
        EnPassant
    };

    enum struct CastleType : std::uint8_t
    {
        Short,
        Long
    };

    [[nodiscard]] constexpr CastleType operator!(CastleType ct)
    {
        return static_cast<CastleType>(static_cast<std::uint8_t>(ct) ^ 1);
    }

    // castling is encoded as a king capturing rook
    // ep is encoded as a normal pawn capture (move.to is empty on the board)
    struct Move
    {
        Square from;
        Square to;
        MoveType type = MoveType::Normal;
        Piece promotedPiece = Piece::None;

        [[nodiscard]] constexpr friend bool operator==(const Move& lhs, const Move& rhs) noexcept
        {
            return lhs.from == rhs.from
                && lhs.to == rhs.to
                && lhs.type == rhs.type
                && lhs.promotedPiece == rhs.promotedPiece;
        }

        [[nodiscard]] constexpr friend bool operator!=(const Move& lhs, const Move& rhs) noexcept
        {
            return !(lhs == rhs);
        }
    };

    static_assert(sizeof(Move) == 4);

    enum struct CastlingRights : std::uint8_t
    {
        None = 0x0,
        WhiteKingSide = 0x1,
        WhiteQueenSide = 0x2,
        BlackKingSide = 0x4,
        BlackQueenSide = 0x8,
        White = WhiteKingSide | WhiteQueenSide,
        Black = BlackKingSide | BlackQueenSide,
        All = WhiteKingSide | WhiteQueenSide | BlackKingSide | BlackQueenSide
    };

    [[nodiscard]] constexpr CastlingRights operator|(CastlingRights lhs, CastlingRights rhs)
    {
        return static_cast<CastlingRights>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
    }

    [[nodiscard]] constexpr CastlingRights operator&(CastlingRights lhs, CastlingRights rhs)
    {
        return static_cast<CastlingRights>(static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
    }

    [[nodiscard]] constexpr CastlingRights operator~(CastlingRights lhs)
    {
        return static_cast<CastlingRights>(~static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(CastlingRights::All));
    }

    constexpr CastlingRights& operator|=(CastlingRights& lhs, CastlingRights rhs)
    {
        lhs = static_cast<CastlingRights>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
        return lhs;
    }

    constexpr CastlingRights& operator&=(CastlingRights& lhs, CastlingRights rhs)
    {
        lhs = static_cast<CastlingRights>(static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
        return lhs;
    }
    // checks whether lhs contains rhs
    [[nodiscard]] constexpr bool contains(CastlingRights lhs, CastlingRights rhs)
    {
        return (lhs & rhs) == rhs;
    }

    struct Board
    {
        Board() noexcept :
            m_pieces{},
            m_pocketCount{},
            m_pieceCount{},
            m_kings{}
        {
            std::fill_n(m_pieces, static_cast<uint8_t>(Square::NB), Piece::None);
        }

        constexpr void place(Piece piece, Square sq)
        {
            if (type_of(piece) == PieceType::King)
            {
                m_kings[static_cast<uint8_t>(color_of(piece))] = sq;
                assert(sq != Square::NB || KING_SQUARES == 1);
                if (sq == Square::NB)
                    // No king
                    return;
            }
            auto oldPiece = m_pieces[static_cast<uint8_t>(sq)];
            m_pieces[static_cast<uint8_t>(sq)] = piece;
            if (oldPiece != Piece::None)
                --m_pieceCount;
            if (piece != Piece::None)
                ++m_pieceCount;
        }

        constexpr void setHandCount(Piece piece, uint8_t count)
        {
            m_pocketCount[static_cast<uint8_t>(piece)] = count;
        }

        constexpr int getHandCount(Piece piece) const
        {
            return m_pocketCount[static_cast<uint8_t>(piece)];
        }

        [[nodiscard]] constexpr Piece pieceAt(Square sq) const
        {
            return m_pieces[static_cast<uint8_t>(sq)];
        }

        [[nodiscard]] inline Square kingSquare(Color c) const
        {
            return m_kings[static_cast<uint8_t>(c)];
        }

        [[nodiscard]] constexpr std::uint8_t pieceCount() const
        {
            return m_pieceCount;
        }

        const Piece* piecesRaw() const;

    private:
        Piece m_pieces[(unsigned int)(Square::NB)];
        uint8_t m_pocketCount[(unsigned int)(Piece::NB)];
        uint8_t m_pieceCount;
        Square m_kings[(unsigned int)(Color::NB)];
    };


    struct Position : public Board
    {
        using BaseType = Board;

        Position() noexcept :
            Board(),
            m_sideToMove(Color::White),
            m_epSquare(Square::NB),
            m_castlingRights(CastlingRights::All),
            m_rule50Counter(0),
            m_ply(0)
        {
        }

        constexpr Position(const Board& board, const Board& pocket, Color sideToMove, Square epSquare, CastlingRights castlingRights) :
            Board(board),
            m_sideToMove(sideToMove),
            m_epSquare(epSquare),
            m_castlingRights(castlingRights),
            m_rule50Counter(0),
            m_ply(0)
        {
        }

        void setEpSquare(Square sq)
        {
            m_epSquare = sq;
        }

        constexpr void setSideToMove(Color color)
        {
            m_sideToMove = color;
        }

        constexpr void addCastlingRights(CastlingRights rights)
        {
            m_castlingRights |= rights;
        }

        constexpr void setCastlingRights(CastlingRights rights)
        {
            m_castlingRights = rights;
        }

        constexpr void setRule50Counter(std::uint8_t v)
        {
            m_rule50Counter = v;
        }

        constexpr void setPly(std::uint16_t ply)
        {
            m_ply = ply;
        }

        inline void setFullMove(std::uint16_t hm)
        {
            m_ply = 2 * hm - 1 + (m_sideToMove == Color::Black);
        }

        [[nodiscard]] constexpr Color sideToMove() const
        {
            return m_sideToMove;
        }

        [[nodiscard]] inline std::uint8_t rule50Counter() const
        {
            return m_rule50Counter;
        }

        [[nodiscard]] inline std::uint16_t ply() const
        {
            return m_ply;
        }

        [[nodiscard]] inline std::uint16_t fullMove() const
        {
            return (m_ply + 1) / 2;
        }

    protected:
        Color m_sideToMove;
        Square m_epSquare;
        CastlingRights m_castlingRights;
        std::uint8_t m_rule50Counter;
        std::uint16_t m_ply;

        static_assert(sizeof(Color) + sizeof(Square) + sizeof(CastlingRights) + sizeof(std::uint8_t) == 4);
    };

}

namespace binpack
{
    constexpr std::size_t KiB = 1024;
    constexpr std::size_t MiB = (1024*KiB);
    constexpr std::size_t GiB = (1024*MiB);

    constexpr std::size_t suggestedChunkSize = MiB;
    constexpr std::size_t maxMovelistSize = 10*KiB; // a safe upper bound
    constexpr std::size_t maxChunkSize = 100*MiB; // to prevent malformed files from causing huge allocations

    using namespace std::literals;

    namespace nodchip
    {
        // This namespace contains modified code from https://github.com/nodchip/Stockfish
        // which is released under GPL v3 license https://www.gnu.org/licenses/gpl-3.0.html

        using namespace std;

        struct StockfishMove
        {
            [[nodiscard]] chess::Move toMove() const
            {
                const chess::Square to = static_cast<chess::Square>((m_raw & (0b111111 << 0) >> 0));
                const chess::Square from = static_cast<chess::Square>((m_raw & (0b111111 << 6)) >> 6);

                const unsigned promotionIndex = (m_raw & (0b11 << 12)) >> 12;
                const chess::PieceType promotionType = static_cast<chess::PieceType>(static_cast<int>(chess::PieceType::Knight) + promotionIndex);

                const unsigned moveFlag = (m_raw & (0b11 << 14)) >> 14;
                chess::MoveType type = chess::MoveType::Normal;
                if (moveFlag == 1) type = chess::MoveType::Promotion;
                else if (moveFlag == 2) type = chess::MoveType::EnPassant;
                else if (moveFlag == 3) type = chess::MoveType::Castle;

                if (type == chess::MoveType::Promotion)
                {
                    const chess::Color stm = rank_of(to) >= chess::Rank::RANK_5 ? chess::Color::White : chess::Color::Black;
                    return chess::Move{from, to, type, make_piece(promotionType, stm)};
                }

                return chess::Move{from, to, type};
            }

        private:
            std::uint16_t m_raw;
        };
        static_assert(sizeof(StockfishMove) == sizeof(std::uint16_t));

        struct PackedSfen
        {
            uint8_t data[64];
        };

        struct PackedSfenValue
        {
            // phase
            PackedSfen sfen;

            // Evaluation value returned from Learner::search()
            int16_t score;

            // PV first move
            // Used when finding the match rate with the teacher
            StockfishMove move;

            // Trouble of the phase from the initial phase.
            uint16_t gamePly;

            // 1 if the player on this side ultimately wins the game. -1 if you are losing.
            // 0 if a draw is reached.
            // The draw is in the teacher position generation command gensfen,
            // Only write if LEARN_GENSFEN_DRAW_RESULT is enabled.
            int8_t game_result;

            // When exchanging the file that wrote the teacher aspect with other people
            //Because this structure size is not fixed, pad it so that it is 40 bytes in any environment.
            uint8_t padding;

            // 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
        };
        static_assert(sizeof(PackedSfenValue) == 72);
        // Class that handles bitstream

        // useful when doing aspect encoding
        struct BitStream
        {
            // Set the memory to store the data in advance.
            // Assume that memory is cleared to 0.
            void  set_data(uint8_t* data_) { data = data_; reset(); }

            // Get the pointer passed in set_data().
            uint8_t* get_data() const { return data; }

            // Get the cursor.
            int get_cursor() const { return bit_cursor; }

            // reset the cursor
            void reset() { bit_cursor = 0; }

            // Get 1 bit from the stream.
            int read_one_bit()
            {
                int b = (data[bit_cursor / 8] >> (bit_cursor & 7)) & 1;
                ++bit_cursor;

                return b;
            }

            // read n bits of data
            // Reverse conversion of write_n_bit().
            int read_n_bit(int n)
            {
                int result = 0;
                for (int i = 0; i < n; ++i)
                    result |= read_one_bit() ? (1 << i) : 0;

                return result;
            }

        private:
            // Next bit position to read/write.
            int bit_cursor;

            // data entity
            uint8_t* data;
        };


        // Huffman coding
        // * is simplified from mini encoding to make conversion easier.
        //
        // Huffman Encoding
        //
        // Empty  xxxxxxx0
        // Pawn   xxxxx001 + 1 bit (Color)
        // Knight xxxxx011 + 1 bit (Color)
        // Bishop xxxxx101 + 1 bit (Color)
        // Rook   xxxxx111 + 1 bit (Color)
        // Queen   xxxx1001 + 1 bit (Color)
        //
        // Worst case:
        // - 32 empty squares    32 bits
        // - 30 pieces           150 bits
        // - 2 kings             12 bits
        // - castling rights     4 bits
        // - ep square           7 bits
        // - rule50              7 bits
        // - game ply            16 bits
        // - TOTAL               228 bits < 256 bits

        struct HuffmanedPiece
        {
            int code; // how it will be coded
            int bits; // How many bits do you have
        };

        // NOTE: Order adjusted for this library because originally NO_PIECE had index 0
        constexpr HuffmanedPiece huffman_table[] =
        {
            {0b00000,1}, // NO_PIECE
            {0b00001,5}, // PAWN
            {0b00011,5}, // KNIGHT
            {0b00101,5}, // BISHOP
            {0b00111,5}, // ROOK
            {0b01001,5}, // QUEEN
            {0b01011,5}, //
            {0b01101,5}, //
            {0b01111,5}, //
            {0b10001,5}, //
            {0b10011,5}, //
            {0b10101,5}, //
            {0b10111,5}, //
            {0b11001,5}, //
            {0b11011,5}, //
            {0b11101,5}, //
            {0b11111,5}, //
        };

        // Class for compressing/decompressing sfen
        // sfen can be packed to 256bit (32bytes) by Huffman coding.
        // This is proven by mini. The above is Huffman coding.
        //
        // Internal format = 1-bit turn + 7-bit king position *2 + piece on board (Huffman coding) + hand piece (Huffman coding)
        // Side to move (White = 0, Black = 1) (1bit)
        // White King Position (6 bits)
        // Black King Position (6 bits)
        // Huffman Encoding of the board
        // Castling availability (1 bit x 4)
        // En passant square (1 or 1 + 6 bits)
        // Rule 50 (6 bits)
        // Game play (8 bits)
        //
        // TODO(someone): Rename SFEN to FEN.
        //
        struct SfenPacker
        {
            // sfen packed by pack() (256bit = 32bytes)
            // Or sfen to decode with unpack()
            uint8_t *data; // uint8_t[32];

            BitStream stream;

            // Read one board piece from stream
            [[nodiscard]] chess::Piece read_board_piece_from_stream()
            {
                int pr = 0;
                int code = 0, bits = 0;
                while (true)
                {
                    code |= stream.read_one_bit() << bits;
                    ++bits;

                    assert(bits <= 6);

                    for (pr = 0; pr <= static_cast<int>(chess::PieceType::None); ++pr)
                        if (huffman_table[pr].code == code
                            && huffman_table[pr].bits == bits)
                            goto Found;
                }
            Found:;
                if (pr == 0)
                    return chess::Piece::None;

                // first and second flag
                chess::Color c = (chess::Color)stream.read_one_bit();

                return make_piece(static_cast<chess::PieceType>(pr - 1), c);
            }
        };


        [[nodiscard]] inline chess::Position pos_from_packed_sfen(const PackedSfen& sfen)
        {
            SfenPacker packer;
            auto& stream = packer.stream;
            stream.set_data(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&sfen)));

            chess::Position pos{};

            // Active color
            pos.setSideToMove((chess::Color)stream.read_one_bit());

            // First the position of the ball
            pos.place(make_piece(chess::PieceType::King, chess::Color::White), static_cast<chess::Square>(stream.read_n_bit(7)));
            pos.place(make_piece(chess::PieceType::King, chess::Color::Black), static_cast<chess::Square>(stream.read_n_bit(7)));

            // Piece placement
            for (chess::Rank r = chess::Rank::RANK_MAX; r >= chess::Rank::RANK_1; --r)
            {
                for (chess::File f = chess::File::FILE_A; f <= chess::File::FILE_MAX; ++f)
                {
                    auto sq = make_square(f, r);

                    if (type_of(pos.pieceAt(sq)) != chess::PieceType::King)
                    {
                        assert(pos.pieceAt(sq) == chess::Piece::None);
                        chess::Piece pc = packer.read_board_piece_from_stream();
                        // There may be no pieces, so skip in that case.
                        if (pc != chess::Piece::None)
                            pos.place(pc, sq);
                    }
                    assert(stream.get_cursor() <= 512);
                }
            }

            for (chess::Color c : { chess::Color::White, chess::Color::Black })
                for (chess::PieceType pt = chess::PieceType::Pawn; pt <= chess::PieceType::MaxPiece; ++pt)
                    pos.setHandCount(make_piece(pt, c), static_cast<int>(stream.read_n_bit(5)));

            // Castling availability.
            chess::CastlingRights cr = chess::CastlingRights::None;
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::WhiteKingSide;
            }
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::WhiteQueenSide;
            }
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::BlackKingSide;
            }
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::BlackQueenSide;
            }
            pos.setCastlingRights(cr);

            // En passant square. Ignore if no pawn capture is possible
            if (stream.read_one_bit()) {
                chess::Square ep_square = static_cast<chess::Square>(stream.read_n_bit(7));
                pos.setEpSquare(ep_square);
            }

            // Halfmove clock
            std::uint8_t rule50 = stream.read_n_bit(6);

            // Fullmove number
            std::uint16_t fullmove = stream.read_n_bit(8);

            // Fullmove number, high bits
            // This was added as a fix for fullmove clock
            // overflowing at 256. This change is backwards compatibile.
            fullmove |= stream.read_n_bit(8) << 8;

            // Read the highest bit of rule50. This was added as a fix for rule50
            // counter having only 6 bits stored.
            // In older entries this will just be a zero bit.
            rule50 |= stream.read_n_bit(1) << 6;

            pos.setFullMove(fullmove);
            pos.setRule50Counter(rule50);

            assert(stream.get_cursor() <= 512);

            return pos;
        }
    }

    struct TrainingDataEntry
    {
        chess::Position pos;
        chess::Move move;
        std::int16_t score;
        std::uint16_t ply;
        std::int16_t result;
    };

    [[nodiscard]] inline TrainingDataEntry packedSfenValueToTrainingDataEntry(const nodchip::PackedSfenValue& psv)
    {
        TrainingDataEntry ret;

        ret.pos = nodchip::pos_from_packed_sfen(psv.sfen);
        ret.move = psv.move.toMove();
        ret.score = psv.score;
        ret.ply = psv.gamePly;
        ret.result = psv.game_result;

        return ret;
    }
}
