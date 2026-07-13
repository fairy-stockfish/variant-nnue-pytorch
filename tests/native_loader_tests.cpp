#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "../training_data_loader.cpp"

using bin::TrainingDataEntry;
using namespace chess;

static Square sq(int file, int rank)
{
    return static_cast<Square>(rank * static_cast<int>(File::FILE_NB) + file);
}

static void set_bits(bin::nodchip::PackedSfen& sfen, int offset, std::uint32_t value, int width)
{
    for (int bit = 0; bit < width; ++bit)
    {
        const auto mask = static_cast<std::uint8_t>(1U << ((offset + bit) & 7));
        auto& byte = sfen.data[(offset + bit) / 8];
        if ((value >> bit) & 1U)
            byte |= mask;
        else
            byte &= static_cast<std::uint8_t>(~mask);
    }
}

static void set_raw_move(bin::nodchip::PackedSfenValue& record, std::uint16_t raw)
{
    static_assert(sizeof(record.move) == sizeof(raw));
    std::memcpy(&record.move, &raw, sizeof(raw));
}

static bin::nodchip::PackedSfenValue valid_legacy_record()
{
    bin::nodchip::PackedSfenValue record{};
    int cursor = 0;
    auto write = [&](std::uint32_t value, int width) {
        set_bits(record.sfen, cursor, value, width);
        cursor += width;
    };

    write(0, 1);   // White to move.
    write(4, 7);   // White king e1.
    write(60, 7);  // Black king e8.

    for (int rank = static_cast<int>(Rank::RANK_MAX);
         rank >= static_cast<int>(Rank::RANK_1);
         --rank)
    {
        for (int file = static_cast<int>(File::FILE_A);
             file <= static_cast<int>(File::FILE_MAX);
             ++file)
        {
            const auto square = sq(file, rank);
            if (square == sq(4, 0) || square == sq(4, 7))
                continue;
            if (square == sq(0, 1))
            {
                write(0b00001, 5);  // Pawn.
                write(0, 1);        // White.
            }
            else
                write(0, 1);        // Empty.
        }
    }

    for (Color color : {Color::White, Color::Black})
        for (PieceType pt = PieceType::Pawn; pt <= PieceType::MaxPiece; ++pt)
        {
            (void) color;
            write(0, DATA_SIZE > 512 ? 7 : 5);
        }

    for (int i = 0; i < 4; ++i)
        write(0, 1);  // No castling rights.
    write(0, 1);      // No en-passant square.
    write(0, 6);      // Rule 50 low bits.
    write(1, 8);      // Fullmove low bits.
    write(0, 8);      // Fullmove high bits.
    write(0, 1);      // Rule 50 high bit.
    assert(cursor <= DATA_SIZE);

    set_raw_move(record, static_cast<std::uint16_t>(16 | (8 << 6)));  // a2a3.
    record.game_result = 0;
    return record;
}

static void expect_invalid_record(
    const bin::nodchip::PackedSfenValue& record,
    const char* expected_message)
{
    try
    {
        (void) bin::packedSfenValueToTrainingDataEntry(record);
        assert(false && "corrupt Legacy72 record was accepted");
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string(error.what()).find(expected_message) != std::string::npos);
    }
}

static TrainingDataEntry quiet_entry()
{
    TrainingDataEntry entry{};
    entry.pos.place(make_piece(PieceType::King, Color::White), sq(4, 0));
    entry.pos.place(make_piece(PieceType::King, Color::Black), sq(4, 7));
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(0, 1));
    entry.pos.setSideToMove(Color::White);
    entry.move = Move{sq(0, 1), sq(0, 2), MoveType::Normal};
    return entry;
}

static void test_smart_filter()
{
    auto entry = quiet_entry();
    assert(!is_smart_filtered(entry));
    auto predicate = make_skip_predicate(true, 0, 0);
    assert(!predicate(entry));

    entry.pos.place(make_piece(PieceType::Knight, Color::Black), sq(0, 2));
    assert(is_smart_filtered(entry));
    assert(predicate(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(6, 6));
    entry.pos.place(make_piece(PieceType::Rook, Color::Black), sq(7, 7));
    entry.move = Move{
        sq(6, 6),
        sq(7, 7),
        MoveType::Promotion,
        make_piece(PieceType::Queen, Color::White)
    };
    assert(is_smart_filtered(entry));

    entry = quiet_entry();
    entry.move = Move{sq(0, 1), sq(1, 2), MoveType::EnPassant};
    assert(is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Rook, Color::White), sq(7, 0));
    entry.move = Move{sq(4, 0), sq(7, 0), MoveType::Castle};
    assert(!is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Pawn, Color::Black), sq(3, 1));
    assert(is_smart_filtered(entry)); // black pawn attacks the white king on e1

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Knight, Color::Black), sq(5, 2));
    assert(is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Rook, Color::Black), sq(4, 5));
    assert(is_smart_filtered(entry));
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(4, 3));
    assert(!is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Bishop, Color::Black), sq(7, 3));
    assert(is_smart_filtered(entry));
    entry.pos.place(make_piece(PieceType::Pawn, Color::White), sq(6, 2));
    assert(!is_smart_filtered(entry));

    entry = quiet_entry();
    entry.pos.place(make_piece(PieceType::Rook, Color::White), sq(4, 6));
    assert(!is_smart_filtered(entry)); // attacking the non-moving king is irrelevant

    entry = TrainingDataEntry{};
    entry.pos.setSideToMove(Color::White);
    entry.move = Move{sq(0, 1), sq(0, 2), MoveType::Normal};
    assert(!is_smart_filtered(entry)); // missing king is handled safely

    entry = quiet_entry();
    entry.pos.place(Piece::None, sq(4, 7));
    entry.pos.place(make_piece(PieceType::King, Color::Black), sq(4, 1));
    // The legacy smart-skip heuristic deliberately uses geometric orthodox
    // check detection; it is not an Atomic legality oracle.
    assert(is_smart_filtered(entry));
}

static void test_seeded_random_skipping()
{
    const auto entry = quiet_entry();
    auto first = make_skip_predicate(false, 3, 0x123456789abcdef0ULL);
    auto second = make_skip_predicate(false, 3, 0x123456789abcdef0ULL);
    auto different = make_skip_predicate(false, 3, 0x0fedcba987654321ULL);

    std::uint64_t first_bits = 0;
    std::uint64_t second_bits = 0;
    std::uint64_t different_bits = 0;
    for (int i = 0; i < 64; ++i)
    {
        first_bits |= static_cast<std::uint64_t>(first(entry)) << i;
        second_bits |= static_cast<std::uint64_t>(second(entry)) << i;
        different_bits |= static_cast<std::uint64_t>(different(entry)) << i;
    }
    assert(first_bits == second_bits);
    assert(first_bits != different_bits);
    assert(first_bits == 0xaf46d7fda3c76fffULL);

    auto maximum = make_skip_predicate(false, std::numeric_limits<int>::max(), 1);
    const auto start = std::chrono::steady_clock::now();
    (void) maximum(entry);
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
}

static void test_c_api_validation()
{
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 0, "unused.bin", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "unused.bin", 0, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "unused.bin", 1, false, false, -1, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed(nullptr, 1, "unused.bin", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("unknown", 1, "unused.bin", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "unused.txt", 1, false, false, 0, 0) == nullptr);
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, "does-not-exist.bin", 1, false, false, 0, 0) == nullptr);

    const auto unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );
    const auto malformed = std::filesystem::temp_directory_path()
        / ("malformed-legacy-loader-test-" + unique_suffix + ".bin");
    {
        std::ofstream output(malformed, std::ios::binary);
        assert(output);
        output.put('\0');
    }
    const auto malformed_string = malformed.string();
    assert(create_sparse_batch_stream_with_seed("HalfKAv2", 1, malformed_string.c_str(), 1, false, false, 0, 0) == nullptr);
    assert(std::filesystem::remove(malformed));
    assert(fetch_next_sparse_batch(nullptr) == nullptr);
}

static void test_corrupt_full_size_records_fail_closed()
{
    const auto valid = valid_legacy_record();
    const auto decoded = bin::packedSfenValueToTrainingDataEntry(valid);
    assert(decoded.move.from == sq(0, 1));
    assert(decoded.move.to == sq(0, 2));

    std::vector<std::pair<std::string, bin::nodchip::PackedSfenValue>> corrupt;

    auto duplicate_king = valid;
    set_bits(duplicate_king.sfen, 8, 4, 7);
    corrupt.emplace_back("same square", duplicate_king);

    auto king_out_of_range = valid;
    set_bits(king_out_of_range.sfen, 1, 65, 7);
    corrupt.emplace_back("outside the board", king_out_of_range);

    auto invalid_huffman = valid;
    set_bits(invalid_huffman.sfen, 15, 0b01011, 5);
    corrupt.emplace_back("Huffman", invalid_huffman);

    auto invalid_move = valid;
    set_raw_move(invalid_move, static_cast<std::uint16_t>(8 | (8 << 6)));
    corrupt.emplace_back("move squares", invalid_move);

    auto invalid_result = valid;
    invalid_result.game_result = 2;
    corrupt.emplace_back("result", invalid_result);

    const auto unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()
    );

    for (std::size_t index = 0; index < corrupt.size(); ++index)
    {
        const auto& [message, record] = corrupt[index];
        expect_invalid_record(record, message.c_str());

        const auto path = std::filesystem::temp_directory_path()
            / ("corrupt-legacy72-" + unique_suffix + "-" + std::to_string(index) + ".bin");
        {
            std::ofstream output(path, std::ios::binary);
            assert(output);
            output.write(reinterpret_cast<const char*>(&record), sizeof(record));
            assert(output);
        }

        const auto path_string = path.string();
        auto stream = create_sparse_batch_stream_with_seed(
            "HalfKAv2", 2, path_string.c_str(), 1, false, false, 0, 0
        );
        assert(stream != nullptr);
        assert(fetch_next_sparse_batch(stream) == nullptr);
        const char* native_error = get_sparse_batch_stream_error(stream);
        assert(native_error != nullptr && native_error[0] != '\0');
        assert(std::string(native_error).find(message) != std::string::npos);
        destroy_sparse_batch_stream(stream);
        assert(std::filesystem::remove(path));
    }
}

int main()
{
    static_assert(sizeof(bin::nodchip::PackedSfenValue) == 72, "legacy ABI must stay 72 bytes");
    test_smart_filter();
    test_seeded_random_skipping();
    test_c_api_validation();
    test_corrupt_full_size_records_fail_closed();
    return 0;
}
