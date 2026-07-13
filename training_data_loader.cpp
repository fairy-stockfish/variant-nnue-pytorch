#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <exception>
#include <iterator>
#include <future>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <random>
#include <stdexcept>
#include <vector>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL
#endif

using namespace bin;
using namespace chess;

static constexpr int MAX_PIECES = PIECE_COUNT;
static constexpr int MAX_HAND_PIECES = POCKETS ? 2 * static_cast<int>(File::FILE_NB) : 0;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return flip_horizontally(flip_vertically(sq));
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (sq == Square::NB)
        // map missing king to zero
        return Square::MIN;
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return flip_vertically(sq);
    }
}

static int map_king(Square sq)
{
    // palace squares for Xiangi/Janggi
    // map accessible king squares skipping the gaps
    if (Square::KNB == Square(9) && Square::KNB != Square::NB)
        return (int(sq) - 6 * (int(sq) / int(File::FILE_NB)) - 3) % int(Square::KNB);

    return int(sq) % int(Square::KNB);
}

struct HalfKP {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = static_cast<int>(PieceType::MaxPiece) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color);
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_KSQ = static_cast<int>(Square::KNB);
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2 - (NUM_KSQ > 1);
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - (NUM_KSQ > 1));
    static constexpr int INPUTS = NUM_PLANES * NUM_KSQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        if (NUM_PT % 2 && p_idx == NUM_PT)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static int feature_index(Color color, Square ksq, int handCount, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return handCount + p_idx * MAX_HAND_PIECES + NUM_SQ * NUM_PT + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    features[j] = feature_index(color, orient_flip(color, ksq), i, make_piece(pt, c));
                    ++j;
                }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - 2 * (HalfKAv2::NUM_KSQ > 1));
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    auto p_idx = static_cast<int>(pt) * 2 + (c != color);
                    features[j] = offset + i + p_idx * MAX_HAND_PIECES + HalfKAv2::NUM_SQ * NUM_PT;
                    ++j;
                }

        return { j, INPUTS };
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = static_cast<int>(entries.size());
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.pieceCount() - 1) * 8 / MAX_PIECES;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, skipPredicate))
    {
        if (!m_stream)
            throw std::invalid_argument("Unsupported training-data file (expected a .bin file)");

        std::ifstream input(filename, std::ios::binary | std::ios::ate);
        if (!input)
            throw std::invalid_argument("Could not open the training-data file");
        const auto size = static_cast<std::streamoff>(input.tellg());
        if (size <= 0 || size % static_cast<std::streamoff>(sizeof(bin::nodchip::PackedSfenValue)) != 0)
            throw std::invalid_argument("Legacy training-data size must be a positive multiple of 72 bytes");
    }

    virtual StorageT* next() = 0;

    void clear_last_error() noexcept
    {
        m_last_error[0] = '\0';
    }

    void set_last_error(const char* error) noexcept
    {
        std::snprintf(
            m_last_error.data(),
            m_last_error.size(),
            "%s",
            error == nullptr ? "unknown native loader error" : error
        );
    }

    [[nodiscard]] const char* last_error() const noexcept
    {
        return m_last_error.data();
    }

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
    std::array<char, 1024> m_last_error{};
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            try
            {
                std::vector<TrainingDataEntry> entries;
                entries.reserve(m_batch_size);

                while(!m_stop_flag.load())
                {
                    entries.clear();
                    std::uint64_t sequence = 0;

                    {
                        std::unique_lock lock(m_stream_mutex);
                        BaseType::m_stream->fill(entries, m_batch_size);
                        if (entries.empty())
                        {
                            break;
                        }
                        sequence = m_next_input_sequence++;
                    }

                    auto batch = std::make_unique<StorageT>(FeatureSet{}, entries);

                    {
                        std::unique_lock lock(m_batch_mutex);
                        m_batches_not_full.wait(lock, [this, sequence]() {
                            // Always allow the next batch the consumer needs. Without
                            // this exception, later batches can fill the bounded map
                            // while the slowest worker is still constructing the
                            // earliest one, deadlocking the whole stream.
                            return m_batches.size() < static_cast<std::size_t>(m_concurrency + 1)
                                || sequence == m_next_output_sequence
                                || m_stop_flag.load();
                        });

                        if (m_stop_flag.load())
                        {
                            break;
                        }

                        const auto [unused, inserted] = m_batches.emplace(sequence, batch.get());
                        if (!inserted)
                            throw std::logic_error("duplicate native-loader batch sequence");
                        batch.release();

                        lock.unlock();
                        m_batches_any.notify_all();
                    }
                }
            }
            catch (...)
            {
                {
                    std::unique_lock lock(m_batch_mutex);
                    if (!m_worker_exception)
                        m_worker_exception = std::current_exception();
                    m_stop_flag.store(true);
                }
                m_batches_not_full.notify_all();
                m_batches_any.notify_all();
            }

            m_num_workers.fetch_sub(1);
            m_batches_any.notify_all();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() {
            return m_batches.find(m_next_output_sequence) != m_batches.end()
                || m_num_workers.load() == 0
                || m_worker_exception;
        });

        if (m_worker_exception)
            std::rethrow_exception(m_worker_exception);

        const auto it = m_batches.find(m_next_output_sequence);
        if (it != m_batches.end())
        {
            auto batch = it->second;
            m_batches.erase(it);
            ++m_next_output_sequence;

            lock.unlock();
            m_batches_not_full.notify_all();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();
        m_batches_any.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& [sequence, batch] : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::map<std::uint64_t, StorageT*> m_batches;
    std::uint64_t m_next_input_sequence{0};
    std::uint64_t m_next_output_sequence{0};
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::exception_ptr m_worker_exception;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers{0};

    std::vector<std::thread> m_workers;
};


static bool is_capture(const TrainingDataEntry& e)
{
    if (e.move.type == MoveType::EnPassant)
        return true;
    if (e.move.type == MoveType::Castle)
        return false;

    const auto from = static_cast<unsigned int>(e.move.from);
    const auto to = static_cast<unsigned int>(e.move.to);
    if (from >= static_cast<unsigned int>(Square::NB)
        || to >= static_cast<unsigned int>(Square::NB))
        return false;

    const auto mover = e.pos.pieceAt(e.move.from);
    const auto target = e.pos.pieceAt(e.move.to);
    return mover != Piece::None
        && target != Piece::None
        && color_of(mover) != color_of(target);
}

static bool square_has_attacker(const Position& pos, Square target, Color attacker)
{
    const int target_square = static_cast<int>(target);
    if (target_square < 0 || target_square >= static_cast<int>(Square::NB))
        return false;

    const int target_file = target_square % static_cast<int>(File::FILE_NB);
    const int target_rank = target_square / static_cast<int>(File::FILE_NB);

    const auto piece_at = [&pos](int file, int rank) {
        if (file < 0 || file >= static_cast<int>(File::FILE_NB)
            || rank < 0 || rank >= static_cast<int>(Rank::RANK_NB))
            return Piece::None;
        return pos.pieceAt(static_cast<Square>(rank * static_cast<int>(File::FILE_NB) + file));
    };

    const auto is_attacker = [attacker](Piece piece, PieceType type) {
        return piece != Piece::None
            && color_of(piece) == attacker
            && type_of(piece) == type;
    };

    // A white pawn attacks one rank upwards; viewed from the target,
    // its source is one rank below (and vice versa for black).
    const int pawn_source_rank = target_rank + (attacker == Color::White ? -1 : 1);
    for (const int df : {-1, 1})
        if (is_attacker(piece_at(target_file + df, pawn_source_rank), PieceType::Pawn))
            return true;

    static constexpr int knight_offsets[][2] = {
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
        {1, -2}, {1, 2}, {2, -1}, {2, 1}
    };
    for (const auto& offset : knight_offsets)
        if (is_attacker(piece_at(target_file + offset[0], target_rank + offset[1]), PieceType::Knight))
            return true;

    for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr)
            if ((df != 0 || dr != 0)
                && is_attacker(piece_at(target_file + df, target_rank + dr), PieceType::King))
                return true;

    const auto attacked_on_ray = [&](int df, int dr, PieceType slider) {
        int file = target_file + df;
        int rank = target_rank + dr;
        while (file >= 0 && file < static_cast<int>(File::FILE_NB)
            && rank >= 0 && rank < static_cast<int>(Rank::RANK_NB))
        {
            const auto piece = piece_at(file, rank);
            if (piece != Piece::None)
                return color_of(piece) == attacker
                    && (type_of(piece) == slider || type_of(piece) == PieceType::Queen);
            file += df;
            rank += dr;
        }
        return false;
    };

    static constexpr int bishop_directions[][2] = {
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
    };
    for (const auto& direction : bishop_directions)
        if (attacked_on_ray(direction[0], direction[1], PieceType::Bishop))
            return true;

    static constexpr int rook_directions[][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}
    };
    for (const auto& direction : rook_directions)
        if (attacked_on_ray(direction[0], direction[1], PieceType::Rook))
            return true;

    return false;
}

bool is_smart_filtered(const TrainingDataEntry& e)
{
    const auto side_to_move = e.pos.sideToMove();
    const auto king_square = e.pos.kingSquare(side_to_move);
    const auto king_index = static_cast<unsigned int>(king_square);
    const bool valid_king = king_index < static_cast<unsigned int>(Square::NB)
        && e.pos.pieceAt(king_square) != Piece::None
        && type_of(e.pos.pieceAt(king_square)) == PieceType::King
        && color_of(e.pos.pieceAt(king_square)) == side_to_move;

    const auto opponent = side_to_move == Color::White ? Color::Black : Color::White;
    return is_capture(e)
        || (valid_king && square_has_attacker(e.pos, king_square, opponent));
}


std::function<bool(const TrainingDataEntry&)> make_skip_predicate(bool filtered, int random_fen_skipping, std::uint64_t seed)
{
    if (filtered || random_fen_skipping)
    {
        return [
            random_fen_skipping,
            filtered,
            generator = std::mt19937_64(seed)
            ](const TrainingDataEntry& e) mutable {

            auto do_skip = [&]() {
                const std::uint64_t bound = static_cast<std::uint64_t>(random_fen_skipping) + 1;
                // Rejection sampling avoids modulo bias and is reproducible
                // across standard-library implementations.
                const std::uint64_t threshold = (0 - bound) % bound;
                std::uint64_t value;
                do
                {
                    value = generator();
                } while (value < threshold);
                return value % bound != 0;
            };

            auto do_filter = [&]() {
                return is_smart_filtered(e);
            };

            return (random_fen_skipping && do_skip()) || (filtered && do_filter());
        };
    }

    return nullptr;
}

extern "C" {

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream_with_seed(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping, std::uint64_t seed)
    {
        if (feature_set_c == nullptr || filename == nullptr)
        {
            fprintf(stderr, "feature_set and filename must not be null\n");
            return nullptr;
        }
        if (concurrency < 1 || batch_size < 1 || random_fen_skipping < 0)
        {
            fprintf(stderr, "Invalid loader configuration: concurrency and batch_size must be positive, random_fen_skipping must be non-negative\n");
            return nullptr;
        }

        try
        {
            auto skipPredicate = make_skip_predicate(filtered, random_fen_skipping, seed);
            std::string_view feature_set(feature_set_c);
            if (feature_set == "HalfKP")
                return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
            if (feature_set == "HalfKP^")
                return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
            if (feature_set == "HalfKA")
                return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
            if (feature_set == "HalfKA^")
                return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
            if (feature_set == "HalfKAv2")
                return new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
            if (feature_set == "HalfKAv2^")
                return new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);

            fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
            return nullptr;
        }
        catch (const std::exception& error)
        {
            fprintf(stderr, "Failed to create sparse batch stream: %s\n", error.what());
            return nullptr;
        }
        catch (...)
        {
            fprintf(stderr, "Failed to create sparse batch stream: unknown error\n");
            return nullptr;
        }
    }

    // Preserve the original ABI for external callers. The Python loader uses
    // the seeded entry point above; legacy callers retain their old behavior.
    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
    {
        return create_sparse_batch_stream_with_seed(feature_set_c, concurrency, filename, batch_size, cyclic, filtered, random_fen_skipping, std::random_device{}());
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        if (stream == nullptr)
            return nullptr;

        stream->clear_last_error();
        try
        {
            return stream->next();
        }
        catch (const std::exception& error)
        {
            stream->set_last_error(error.what());
            fprintf(stderr, "Failed to fetch sparse batch: %s\n", error.what());
            return nullptr;
        }
        catch (...)
        {
            stream->set_last_error("unknown native loader error");
            fprintf(stderr, "Failed to fetch sparse batch: unknown error\n");
            return nullptr;
        }
    }

    EXPORT const char* CDECL get_sparse_batch_stream_error(Stream<SparseBatch>* stream)
    {
        return stream == nullptr ? "null native loader stream" : stream->last_error();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}

#ifdef TRAINING_DATA_LOADER_BENCH
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream_with_seed("HalfKP", 4, "10m_d3_q_2.bin", 8192, true, false, 0, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
#endif
