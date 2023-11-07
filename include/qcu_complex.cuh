#pragma once

#include <cstdio>

namespace qcu{

  class Complex {
  private:
    double real_;
    double imag_;

  public:
    __device__ __host__ __forceinline__ Complex(double real, double imag) : real_(real), imag_(imag) {}
    Complex() = default;
    __device__ __host__ __forceinline__ Complex(const Complex &complex) : real_(complex.real_), imag_(complex.imag_) {}
    __device__ __host__ __forceinline__ double norm2() { return sqrt(real_ * real_ + imag_ * imag_); }
    __device__ __host__ __forceinline__ void setImag(double imag) { imag_ = imag; }
    __device__ __host__ __forceinline__ void setReal(double real) { real_ = real; }
    __device__ __host__ __forceinline__ double real() const { return real_; }
    __device__ __host__ __forceinline__ double imag() const { return imag_; }

    __device__ __host__ __forceinline__ Complex &operator=(const Complex &complex)
    {
      real_ = complex.real_;
      imag_ = complex.imag_;
      return *this;
    }
    __device__ __host__ __forceinline__ Complex &operator=(double rhs)
    {
      real_ = rhs;
      imag_ = 0;
      return *this;
    }
    __device__ __host__ __forceinline__ Complex operator+(const Complex &complex) const
    {
      return Complex(real_ + complex.real_, imag_ + complex.imag_);
    }
    __device__ __host__ __forceinline__ Complex operator-(const Complex &complex) const
    {
      return Complex(real_ - complex.real_, imag_ - complex.imag_);
    }
    __device__ __host__ __forceinline__ Complex operator-() const { return Complex(-real_, -imag_); }
    __device__ __host__ __forceinline__ Complex operator*(const Complex &rhs) const
    {
      return Complex(real_ * rhs.real_ - imag_ * rhs.imag_, real_ * rhs.imag_ + imag_ * rhs.real_);
    }
    __device__ __host__ __forceinline__ Complex operator*(const double &rhs) const
    {
      return Complex(real_ * rhs, imag_ * rhs);
    }
    __device__ __host__ Complex &operator*=(const Complex &rhs)
    {
      double real = real_ * rhs.real_ - imag_ * rhs.imag_;
      double imag = real_ * rhs.imag_ + imag_ * rhs.real_;
      this->real_ = real;
      this->imag_ = imag;
      return *this;
    }
    __device__ __host__ __forceinline__ Complex &operator*=(const double &rhs)
    {
      real_ = real_ * rhs;
      imag_ = imag_ * rhs;
      return *this;
    }
    __device__ __host__ __forceinline__ Complex operator/(const double &rhs) { return Complex(real_ / rhs, imag_ / rhs); }
    __device__ __host__ __forceinline__ Complex operator/(const Complex &rhs) const {
      return (*this * rhs.conj()) / (rhs.real()*rhs.real() + rhs.imag()*rhs.imag());
    }
    __device__ __host__ __forceinline__ Complex& operator/=(const Complex &rhs) {
      double new_real = (real_*rhs.real() + imag_*rhs.imag()) / (rhs.real()*rhs.real() + rhs.imag()*rhs.imag());
      double new_imag = (rhs.real()*imag_ - real_*rhs.imag()) / (rhs.real()*rhs.real() + rhs.imag()*rhs.imag());
      real_ = new_real;
      imag_ = new_imag;
      return *this;
    }
    __device__ __host__ __forceinline__ Complex &operator+=(const Complex &rhs)
    {
      real_ += rhs.real_;
      imag_ += rhs.imag_;
      return *this;
    }

    __device__ __host__ __forceinline__ Complex &operator-=(const Complex &rhs)
    {
      real_ -= rhs.real_;
      imag_ -= rhs.imag_;
      return *this;
    }

    __device__ __host__ __forceinline__ Complex &clear2Zero()
    {
      real_ = 0;
      imag_ = 0;
      return *this;
    }


    __device__ __host__ __forceinline__ Complex multiply_i()
    {
      return Complex(-imag_, real_);
    }
    __device__ __host__ __forceinline__ Complex multiply_minus_i()
    {
      return Complex(imag_, -real_);
    }
    __device__ __host__ __forceinline__ Complex &self_multiply_i()
    {
      double temp = real_;
      real_ = -imag_;
      imag_ = temp;
      return *this;
    }
    __device__ __host__ __forceinline__ Complex &self_multiply_minus_i()
    {
      double temp = -real_;
      real_ = imag_;
      imag_ = temp;
      return *this;
    }

    __device__ __host__ __forceinline__ Complex conj() const { return Complex(real_, -imag_); }
    __device__ __host__ __forceinline__ bool operator==(const Complex &rhs) { return real_ == rhs.real_ && imag_ == rhs.imag_; }
    __device__ __host__ __forceinline__ bool operator!=(const Complex &rhs) { return real_ != rhs.real_ || imag_ != rhs.imag_; }
    __device__ __host__ __forceinline__ void output() const { printf("(%lf + %lfi)", real_, imag_); }
  };

};