#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <random>
#include <string>
#include <numeric>
#include <set>
#include <deque>
#include <chrono>
#include <algorithm>
#include <iomanip>
using namespace std;

const int KB_LIMIT = 750;
const int GEN_LEN = 140;
const int EPOCHS = 2;
const int BATCH_SIZE = 32;  // Added missing constant
const double LEARNING_RATE = 0.01;  // Added missing constant

template <typename T>
class Tensor {
private:
    vector<size_t> dims;
    vector<T> data;

    size_t get_flat_index(const vector<size_t>& indices) const {
        size_t flat_idx = 0;
        size_t stride = 1;
        for (int i = dims.size() - 1; i >= 0; --i) {
            flat_idx += indices[i] * stride;
            stride *= dims[i];
        }
        return flat_idx;
    }

public:
    Tensor() = default;

    Tensor(const vector<size_t>& dimensions) : dims(dimensions) {
        size_t size = 1;
        for (size_t dim : dims) size *= dim;
        data.resize(size);
    }

    Tensor(const vector<size_t>& dimensions, T initial_value) : dims(dimensions) {
        size_t size = 1;
        for (size_t dim : dims) size *= dim;
        data.resize(size, initial_value);
    }

    T& at(const vector<size_t>& indices) {
        return data[get_flat_index(indices)];
    }

    const T& at(const vector<size_t>& indices) const {
        return data[get_flat_index(indices)];
    }

    void fillValue(T value) {
        std::fill(data.begin(), data.end(), value);
    }

    size_t size() const {
        return data.size();
    }

    const vector<size_t>& shape() const {
        return dims;
    }

    vector<T>& raw_data() {
        return data;
    }

    const vector<T>& raw_data() const {
        return data;
    }

    // Added static softmax implementation
    static Tensor<T> softmax(const Tensor<T>& input) {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < input.shape()[0]; ++i) {
            T max_val = input.at({i, 0});
            for (size_t j = 1; j < input.shape()[1]; ++j) {
                max_val = max(max_val, input.at({i, j}));
            }

            T sum = 0;
            for (size_t j = 0; j < input.shape()[1]; ++j) {
                output.at({i, j}) = exp(input.at({i, j}) - max_val);
                sum += output.at({i, j});
            }

            for (size_t j = 0; j < input.shape()[1]; ++j) {
                output.at({i, j}) /= sum;
            }
        }
        return output;
    }

    Tensor<T> matmul(const Tensor<T>& other) const {
        vector<size_t> result_dims = {dims[0], other.dims[1]};
        Tensor<T> result(result_dims, 0);

        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < other.dims[1]; ++j) {
                T sum = 0;
                for (size_t k = 0; k < dims[1]; ++k) {
                    sum += at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }

    Tensor<T> transpose() const {
        vector<size_t> transposed_dims = {dims[1], dims[0]};
        Tensor<T> result(transposed_dims);

        for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
                result.at({j, i}) = at({i, j});
            }
        }
        return result;
    }

    void add_grad_(const Tensor<T>& grad, T learning_rate) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += learning_rate * grad.data[i];
        }
    }
};

class TextGenerator {
private:
    unordered_map<string, int> word_to_index;
    unordered_map<int, string> index_to_word;
    int vocab_size;
    const int sequence_length;
    const int embedding_dim;
    Tensor<double> embedding_matrix;
    Tensor<double> projection_matrix;
    mt19937 rng;

    string preprocess_word(const string& word) {  // Added missing function
        string processed = word;
        transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
        return processed;
    }

    Tensor<double> create_input_tensor(const vector<string>& sequence) {
        Tensor<double> input({1, sequence_length * embedding_dim});

        for (int i = 0; i < sequence_length; ++i) {
            auto it = word_to_index.find(sequence[i]);
            if (it != word_to_index.end()) {
                int word_idx = it->second;
                for (int j = 0; j < embedding_dim; ++j) {
                    input.at({0, i * embedding_dim + j}) =
                        embedding_matrix.at({word_idx, j});
                }
            }
        }
        return input;
    }

public:
    TextGenerator(const string& filename, int hidden_size) :
        sequence_length(3),
        embedding_dim(hidden_size),
        rng(chrono::steady_clock::now().time_since_epoch().count())
    {
        build_vocabulary(filename);

        normal_distribution<double> dist(0.0, 1.0 / sqrt(embedding_dim));
        embedding_matrix = Tensor<double>({vocab_size, embedding_dim});
        for (size_t i = 0; i < embedding_matrix.size(); ++i) {
            embedding_matrix.raw_data()[i] = dist(rng);
        }

        projection_matrix = Tensor<double>({embedding_dim * sequence_length, vocab_size});
        for (size_t i = 0; i < projection_matrix.size(); ++i) {
            projection_matrix.raw_data()[i] = dist(rng);
        }
    }

    void build_vocabulary(const string& filename) {
        ifstream file(filename);
        if (!file) {
            throw runtime_error("Could not open file: " + filename);
        }

        set<string> unique_words;
        string word;
        while (file >> word && unique_words.size() < KB_LIMIT) {
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            unique_words.insert(word);
        }

        int index = 0;
        for (const auto& word : unique_words) {
            word_to_index[word] = index;
            index_to_word[index] = word;
            index++;
        }
        vocab_size = index;
    }

    void train(const string& filename, int epochs = EPOCHS, double learning_rate = LEARNING_RATE) {
        cout << "Starting training with:" << endl
             << "- Epochs: " << epochs << endl
             << "- Learning rate: " << learning_rate << endl
             << "- Batch size: " << BATCH_SIZE << endl;

        vector<string> words;
        ifstream file(filename);
        if (!file) {
            throw runtime_error("Could not open file: " + filename);
        }

        string word;
        while (file >> word && words.size() < KB_LIMIT) {
            words.push_back(preprocess_word(word));
        }

        vector<vector<string>> input_sequences;
        vector<string> target_words;

        for (size_t i = 0; i < words.size() - sequence_length; ++i) {
            vector<string> sequence(words.begin() + i, words.begin() + i + sequence_length);
            input_sequences.push_back(sequence);
            target_words.push_back(words[i + sequence_length]);
        }

        const int num_batches = (input_sequences.size() + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            int correct_predictions = 0;

            vector<size_t> indices(input_sequences.size());
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), rng);

            for (int batch = 0; batch < num_batches; ++batch) {
                // Process batch
                vector<vector<string>> batch_sequences;
                vector<string> batch_targets;

                size_t start_idx = batch * BATCH_SIZE;
                size_t end_idx = min(start_idx + BATCH_SIZE, input_sequences.size());

                for (size_t i = start_idx; i < end_idx; ++i) {
                    batch_sequences.push_back(input_sequences[indices[i]]);
                    batch_targets.push_back(target_words[indices[i]]);
                }

                // Forward pass
                Tensor<double> batch_input({batch_sequences.size(), sequence_length * embedding_dim});
                Tensor<double> target_tensor({batch_sequences.size(), vocab_size}, 0.0);

                // Prepare tensors
                for (size_t i = 0; i < batch_sequences.size(); ++i) {
                    const vector<string>& sequence = batch_sequences[i];
                    for (size_t j = 0; j < sequence_length; ++j) {
                        auto it = word_to_index.find(sequence[j]);
                        if (it != word_to_index.end()) {
                            int word_idx = it->second;
                            for (int k = 0; k < embedding_dim; ++k) {
                                batch_input.at({i, j * embedding_dim + k}) =
                                    embedding_matrix.at({word_idx, k});
                            }
                        }
                    }

                    auto target_it = word_to_index.find(batch_targets[i]);
                    if (target_it != word_to_index.end()) {
                        target_tensor.at({i, target_it->second}) = 1.0;
                    }
                }

                // Forward pass
                Tensor<double> hidden = batch_input.matmul(projection_matrix);
                Tensor<double> output = Tensor<double>::softmax(hidden);

                // Calculate metrics
                double batch_loss = 0.0;
                for (size_t i = 0; i < batch_sequences.size(); ++i) {
                    auto target_it = word_to_index.find(batch_targets[i]);
                    if (target_it != word_to_index.end()) {
                        batch_loss -= log(output.at({i, target_it->second}) + 1e-10);

                        int predicted_idx = 0;
                        double max_prob = output.at({i, 0});
                        for (int j = 1; j < vocab_size; ++j) {
                            if (output.at({i, j}) > max_prob) {
                                max_prob = output.at({i, j});
                                predicted_idx = j;
                            }
                        }
                        if (predicted_idx == target_it->second) {
                            correct_predictions++;
                        }
                    }
                }
                total_loss += batch_loss;

                // Backward pass
                Tensor<double> grad_output = output;
                for (size_t i = 0; i < batch_sequences.size(); ++i) {
                    auto target_it = word_to_index.find(batch_targets[i]);
                    if (target_it != word_to_index.end()) {
                        grad_output.at({i, target_it->second}) -= 1.0;
                    }
                }

                // Update weights
                Tensor<double> grad_projection = batch_input.transpose().matmul(grad_output);
                projection_matrix.add_grad_(grad_projection, -learning_rate);

                Tensor<double> grad_input = grad_output.matmul(projection_matrix.transpose());
                for (size_t i = 0; i < batch_sequences.size(); ++i) {
                    const vector<string>& sequence = batch_sequences[i];
                    for (size_t j = 0; j < sequence_length; ++j) {
                        auto it = word_to_index.find(sequence[j]);
                        if (it != word_to_index.end()) {
                            int word_idx = it->second;
                            for (int k = 0; k < embedding_dim; ++k) {
                                embedding_matrix.at({word_idx, k}) -=
                                    learning_rate * grad_input.at({i, j * embedding_dim + k});
                            }
                        }
                    }
                }

                if ((batch + 1) % (num_batches / 10) == 0) {
                    cout << "." << flush;
                }
            }

            double avg_loss = total_loss / input_sequences.size();
            double accuracy = static_cast<double>(correct_predictions) / input_sequences.size();
            cout << "\nEpoch " << epoch + 1 << "/" << epochs
                 << " - Loss: " << fixed << setprecision(4) << avg_loss
                 << " - Accuracy: " << setprecision(2) << accuracy * 100 << "%" << endl;
        }
    }

    Tensor<double> forward(const Tensor<double>& input) {
        Tensor<double> output = input.matmul(projection_matrix);
        return Tensor<double>::softmax(output);
    }
    string generate(const string& seed_text, double temperature = 0.8)
    {
        vector<string> tokens;
        stringstream ss(seed_text);
        string token;
        while (ss >> token)
        {
            transform(token.begin(), token.end(), token.begin(), ::tolower);
            tokens.push_back(token);
        }

        if (tokens.size() < sequence_length)
        {
            throw runtime_error("Seed text must contain at least " +
                                to_string(sequence_length) + " words");
        }

        stringstream result;
        result << seed_text;

        deque<string> current_sequence(tokens.end() - sequence_length, tokens.end());

        for (int i = 0; i < GEN_LEN; ++i)
        {
            // Create input tensor from current sequence
            vector<string> seq_vec(current_sequence.begin(), current_sequence.end());
            Tensor<double> input = create_input_tensor(seq_vec);

            // Get probability distribution
            Tensor<double> output = forward(input);

            // Apply temperature
            vector<double> probs(vocab_size);
            for (int j = 0; j < vocab_size; ++j)
            {
                probs[j] = pow(output.at({0, j}), 1.0 / temperature);
            }

            // Normalize probabilities
            double sum = accumulate(probs.begin(), probs.end(), 0.0);
            for (double& p : probs) p /= sum;

            // Sample next word
            discrete_distribution<> dist(probs.begin(), probs.end());
            int next_word_idx = dist(rng);
            string next_word = index_to_word[next_word_idx];

            result << " " << next_word;

            current_sequence.pop_front();
            current_sequence.push_back(next_word);
        }

        return result.str();
    }
};

int main() {
    try {
        string filename = "test.txt";
        int hidden_size = 256;
        TextGenerator generator(filename, hidden_size);

        // *** TRAINING ADDED HERE ***
        cout << "\nStarting Training...\n";
        generator.train(filename); // Train the model
        cout << "\nTraining Complete.\n";

        cout << "\nEnter a seed text (at least 3 words) or type 'exit' to quit:" << endl;
        string user_input;

        while (true) {
            cout << "> ";
            getline(cin, user_input);

            if (user_input == "exit") break;

            try {
                string generated_text = generator.generate(user_input);
                cout << "Generated text: " << generated_text << endl;
            } catch (const exception& e) {
                cout << "Error: " << e.what() << endl;
            }
        }
    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
