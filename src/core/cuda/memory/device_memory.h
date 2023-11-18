#ifndef D655B777_E51D_4A48_B563_EF9B88B7FC8E
#define D655B777_E51D_4A48_B563_EF9B88B7FC8E

/*
@brief: Dynamic vector implementation for device memory
@file: device_vector.h
*/

/*modifed the gpu memory from the tiny cuda nn.*/
/* 
* Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
* FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "common.h"

PE_BEGIN

// tracks the bytes allcoated by all the memory_t allocations
inline std::atomic<size_t>& total_n_bytes_allocated() {
	static std::atomic<size_t> s_total_n_bytes_allocated{ 0 };
	return s_total_n_bytes_allocated;
}

template <typename U>
class memory_t {

private:
  U* device_ptr = nullptr;
  size_t m_size = 0;

public:

  memory_t() {} 

  memory_t(size_t s) {
    resize(s);
  }

  memory_t<U>& operator=(memory_t<U> obj) {
    std::swap(device_ptr, obj.device_ptr);
    std::swap(m_size, obj.m_size);
    return *this;
  }

  void operator=(memory_t<U>&& obj) {
    *this = move(obj);
  }

  memory_t(memory_t<U>& obj) {
    std::swap(device_ptr, obj.device_ptr);
    std::swap(m_size, obj.m_size);
  }

  explicit memory_t(const memory_t<U>& obj) {
		copy_from_device(obj);
	}

  U* data() {
		return device_ptr;
	}

  size_t get_num_elements() const {
		return m_size;
	}

	size_t size() const {
		return get_num_elements();
	}


	size_t get_bytes() const {
		return m_size * sizeof(U);
	}

	size_t bytes() const {
		return get_bytes();
	}

  void allocate_memory(size_t n_bytes) {
		
		if (n_bytes == 0) {
			return;
		}
		uint8_t* raw_ptr;
		cudaMalloc(&raw_ptr, n_bytes);
		device_ptr = (U*)raw_ptr;
		total_n_bytes_allocated() += n_bytes; 
	}

	void free_memory() {
		if (!device_ptr) {
			return;
		}

		uint8_t* raw_ptr = (uint8_t*)device_ptr;
		cudaFree(raw_ptr);
		total_n_bytes_allocated() -= get_bytes();

		device_ptr = nullptr;
		m_size = 0;
	}

  PE_HOST_DEVICE ~memory_t() {
  #ifndef __CUDA_ARCH__
		try {
			if (device_ptr) {
				free_memory();
				m_size = 0;
			}
		}
		catch (std::runtime_error e) {
			if (std::string{ e.what() }.find("driver shutting down") == std::string::npos) {
				std::cerr << "Could not free memory: " << e.what() << std::endl;
			}
		}
  #endif
	}

	void resize(size_t s) {
		if (m_size != s) {
			if (m_size) {
				try {
					free_memory();
				}
				catch (std::runtime_error e) {
					std::string str(e.what());
					throw std::runtime_error{ "Could not allocate memory: " + str };
				}
			}
			m_size = s;
		}
	}

	void enlarge(const size_t s) {
		if (s > m_size) {
			resize(s);
		}
	}

  void memset(const int value, const size_t num_elements, const size_t offset = 0) {
		if (num_elements + offset > m_size) {
			throw std::runtime_error{ "Could not set memory : Number of elements is larger than allocated memory"};
		}

		cudaMemset(device_ptr + offset, value, num_elements * sizeof(U));
	}


	void memset(const int value) {
		memset(value, m_size);
	}

  void copy_from_host(const U* host_data, const size_t num_elements) {
		cudaMemcpy(data(), host_data, num_elements * sizeof(U), PE_HTD);
	}

  void copy_from_host(const std::vector<U>& data, const size_t num_elements) {
		if (data.m_size() < num_elements) {
			throw std::runtime_error{"Trying to copy " + std::to_string(num_elements) + "but vector m_size is " + std::to_string(data.m_size())};
		}

		copy_from_host(data.data(), num_elements);
	}

	void copy_from_host(const U* data) {
		copy_from_host(data, m_size);
	}

  void copy_from_host(const std::vector<U>& data) {
		if (data.m_size() < m_size) {
			throw std::runtime_error{ "Trying to copy " + std::to_string(m_size) + ", but vector m_size is only " + std::to_string(data.m_size()) + "."};
		}
		copy_from_host(data.data(), m_size);
	}

  void enlarge_and_copy_from_host(const U* data, const size_t num_elements) {
		enlarge(num_elements);
		copy_from_host(data, num_elements);
	}

	void enlarge_and_copy_from_host(const std::vector<U>& data, const size_t num_elements) {
		enlarge_and_copy_from_host(data.data(), num_elements);
	}

	void enlarge_and_copy_from_host(const std::vector<U>& data) {
		enlarge_and_copy_from_host(data.data(), data.m_size());
	}

	void resize_and_copy_from_host(const U* data, const size_t num_elements) {
		resize(num_elements);
		copy_from_host(data, num_elements);
	}

  void resize_and_copy_from_host(const std::vector<U>& data, const size_t num_elements) {
		resize_and_copy_from_host(data.data(), num_elements);
	}

	void resize_and_copy_from_host(const std::vector<U>& data) {
		resize_and_copy_from_host(data.data(), data.m_size());
	}

  void copy_to_host(U* host_data, const size_t num_elements) const {
		if (num_elements > m_size) {
			throw std::runtime_error{ "Trying to copy " + std::to_string(num_elements) + ", but vector m_size is only " + std::to_string(m_size) + "." };
		}
		cudaMemcpy(host_data, device_ptr, num_elements * sizeof(U), PE_DTH);
	}

	void copy_to_host(std::vector<U>& data, const size_t num_elements) const {
		if (data.size() < num_elements) {
			throw std::runtime_error{"Trying to copy " + std::to_string(num_elements) + " elements, but vector m_size is only " + std::to_string(data.size())};
		}
		copy_to_host(data.data(), num_elements);
	}

	void copy_to_host(U* data) const {
		copy_to_host(data, m_size);
	}

	void copy_to_host(std::vector<U>& data) const {
		if (data.size() < m_size) {
			throw std::runtime_error{ "Trying to copy " + std::to_string(m_size) + " elements, but vector m_size is only " + std::to_string(data.size()) };
		}

		copy_to_host(data.data(), m_size);
	}

	void copy_from_device(memory_t<U> obj, const size_t size) {

		if (size == 0) {
			return;
		}

		if (m_size < size) {
			resize(size);
		}

		cudaMemcpy(device_ptr, obj.device_ptr, size * sizeof(U), PE_DTD);
	}

	void copy_to_device(U*& obj, const size_t size) {

		if (size == 0) {
			return;
		}

		cudaMemcpy(device_ptr, obj, size * sizeof(U), PE_DTD);
	}

  void copy_to_device(memory_t<U>& obj) {
		copy_to_device(obj, obj.m_size);
	}

  memory_t<U> copy(size_t size) const {
		memory_t<U> result{ size };
		result.copy_from_device(*this);
		return result;
	}

	memory_t<U> copy() {
		return copy(m_size);
	}

  PE_HOST_DEVICE U& operator[](size_t idx) const {
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %zu\n", idx);
		}

		return device_ptr[idx];
	}

	PE_HOST_DEVICE U& operator[](uint32_t idx) const {
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %u\n", idx);
		}

		return device_ptr[idx];
	}

};

PE_END

#endif /* D655B777_E51D_4A48_B563_EF9B88B7FC8E */
