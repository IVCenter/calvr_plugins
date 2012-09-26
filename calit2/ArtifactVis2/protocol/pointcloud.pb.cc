// Generated by the protocol buffer compiler.  DO NOT EDIT!

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "protocol/pointcloud.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace RemoteKinect {

namespace {

const ::google::protobuf::Descriptor* Point_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Point_reflection_ = NULL;
const ::google::protobuf::Descriptor* PointCloud_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  PointCloud_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_protocol_2fpointcloud_2eproto() {
  protobuf_AddDesc_protocol_2fpointcloud_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "protocol/pointcloud.proto");
  GOOGLE_CHECK(file != NULL);
  Point_descriptor_ = file->message_type(0);
  static const int Point_offsets_[6] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, x_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, y_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, z_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, r_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, g_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, b_),
  };
  Point_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Point_descriptor_,
      Point::default_instance_,
      Point_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Point, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Point));
  PointCloud_descriptor_ = file->message_type(1);
  static const int PointCloud_offsets_[3] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PointCloud, source_serial_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PointCloud, tick_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PointCloud, points_),
  };
  PointCloud_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      PointCloud_descriptor_,
      PointCloud::default_instance_,
      PointCloud_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PointCloud, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(PointCloud, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(PointCloud));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_protocol_2fpointcloud_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Point_descriptor_, &Point::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    PointCloud_descriptor_, &PointCloud::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_protocol_2fpointcloud_2eproto() {
  delete Point::default_instance_;
  delete Point_reflection_;
  delete PointCloud::default_instance_;
  delete PointCloud_reflection_;
}

void protobuf_AddDesc_protocol_2fpointcloud_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\031protocol/pointcloud.proto\022\014RemoteKinec"
    "t\"I\n\005Point\022\t\n\001x\030\001 \002(\002\022\t\n\001y\030\002 \002(\002\022\t\n\001z\030\003 "
    "\002(\002\022\t\n\001r\030\004 \001(\r\022\t\n\001g\030\005 \001(\r\022\t\n\001b\030\006 \001(\r\"V\n\n"
    "PointCloud\022\025\n\rsource_serial\030\001 \002(\t\022\014\n\004tic"
    "k\030\002 \002(\r\022#\n\006points\030\003 \003(\0132\023.RemoteKinect.P"
    "oint", 204);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "protocol/pointcloud.proto", &protobuf_RegisterTypes);
  Point::default_instance_ = new Point();
  PointCloud::default_instance_ = new PointCloud();
  Point::default_instance_->InitAsDefaultInstance();
  PointCloud::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_protocol_2fpointcloud_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_protocol_2fpointcloud_2eproto {
  StaticDescriptorInitializer_protocol_2fpointcloud_2eproto() {
    protobuf_AddDesc_protocol_2fpointcloud_2eproto();
  }
} static_descriptor_initializer_protocol_2fpointcloud_2eproto_;


// ===================================================================

#ifndef _MSC_VER
const int Point::kXFieldNumber;
const int Point::kYFieldNumber;
const int Point::kZFieldNumber;
const int Point::kRFieldNumber;
const int Point::kGFieldNumber;
const int Point::kBFieldNumber;
#endif  // !_MSC_VER

Point::Point()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void Point::InitAsDefaultInstance() {
}

Point::Point(const Point& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void Point::SharedCtor() {
  _cached_size_ = 0;
  x_ = 0;
  y_ = 0;
  z_ = 0;
  r_ = 0u;
  g_ = 0u;
  b_ = 0u;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Point::~Point() {
  SharedDtor();
}

void Point::SharedDtor() {
  if (this != default_instance_) {
  }
}

void Point::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Point::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Point_descriptor_;
}

const Point& Point::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_protocol_2fpointcloud_2eproto();  return *default_instance_;
}

Point* Point::default_instance_ = NULL;

Point* Point::New() const {
  return new Point;
}

void Point::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    x_ = 0;
    y_ = 0;
    z_ = 0;
    r_ = 0u;
    g_ = 0u;
    b_ = 0u;
  }
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Point::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required float x = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &x_)));
          set_has_x();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(21)) goto parse_y;
        break;
      }
      
      // required float y = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32) {
         parse_y:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &y_)));
          set_has_y();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(29)) goto parse_z;
        break;
      }
      
      // required float z = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED32) {
         parse_z:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &z_)));
          set_has_z();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(32)) goto parse_r;
        break;
      }
      
      // optional uint32 r = 4;
      case 4: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_r:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &r_)));
          set_has_r();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(40)) goto parse_g;
        break;
      }
      
      // optional uint32 g = 5;
      case 5: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_g:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &g_)));
          set_has_g();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(48)) goto parse_b;
        break;
      }
      
      // optional uint32 b = 6;
      case 6: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_b:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &b_)));
          set_has_b();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void Point::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required float x = 1;
  if (has_x()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(1, this->x(), output);
  }
  
  // required float y = 2;
  if (has_y()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(2, this->y(), output);
  }
  
  // required float z = 3;
  if (has_z()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(3, this->z(), output);
  }
  
  // optional uint32 r = 4;
  if (has_r()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(4, this->r(), output);
  }
  
  // optional uint32 g = 5;
  if (has_g()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(5, this->g(), output);
  }
  
  // optional uint32 b = 6;
  if (has_b()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(6, this->b(), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* Point::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required float x = 1;
  if (has_x()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(1, this->x(), target);
  }
  
  // required float y = 2;
  if (has_y()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(2, this->y(), target);
  }
  
  // required float z = 3;
  if (has_z()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(3, this->z(), target);
  }
  
  // optional uint32 r = 4;
  if (has_r()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(4, this->r(), target);
  }
  
  // optional uint32 g = 5;
  if (has_g()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(5, this->g(), target);
  }
  
  // optional uint32 b = 6;
  if (has_b()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(6, this->b(), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int Point::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required float x = 1;
    if (has_x()) {
      total_size += 1 + 4;
    }
    
    // required float y = 2;
    if (has_y()) {
      total_size += 1 + 4;
    }
    
    // required float z = 3;
    if (has_z()) {
      total_size += 1 + 4;
    }
    
    // optional uint32 r = 4;
    if (has_r()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->r());
    }
    
    // optional uint32 g = 5;
    if (has_g()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->g());
    }
    
    // optional uint32 b = 6;
    if (has_b()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->b());
    }
    
  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Point::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Point* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Point*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Point::MergeFrom(const Point& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_x()) {
      set_x(from.x());
    }
    if (from.has_y()) {
      set_y(from.y());
    }
    if (from.has_z()) {
      set_z(from.z());
    }
    if (from.has_r()) {
      set_r(from.r());
    }
    if (from.has_g()) {
      set_g(from.g());
    }
    if (from.has_b()) {
      set_b(from.b());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Point::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Point::CopyFrom(const Point& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Point::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000007) != 0x00000007) return false;
  
  return true;
}

void Point::Swap(Point* other) {
  if (other != this) {
    std::swap(x_, other->x_);
    std::swap(y_, other->y_);
    std::swap(z_, other->z_);
    std::swap(r_, other->r_);
    std::swap(g_, other->g_);
    std::swap(b_, other->b_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Point::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Point_descriptor_;
  metadata.reflection = Point_reflection_;
  return metadata;
}


// ===================================================================

#ifndef _MSC_VER
const int PointCloud::kSourceSerialFieldNumber;
const int PointCloud::kTickFieldNumber;
const int PointCloud::kPointsFieldNumber;
#endif  // !_MSC_VER

PointCloud::PointCloud()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void PointCloud::InitAsDefaultInstance() {
}

PointCloud::PointCloud(const PointCloud& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void PointCloud::SharedCtor() {
  _cached_size_ = 0;
  source_serial_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  tick_ = 0u;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

PointCloud::~PointCloud() {
  SharedDtor();
}

void PointCloud::SharedDtor() {
  if (source_serial_ != &::google::protobuf::internal::kEmptyString) {
    delete source_serial_;
  }
  if (this != default_instance_) {
  }
}

void PointCloud::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* PointCloud::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return PointCloud_descriptor_;
}

const PointCloud& PointCloud::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_protocol_2fpointcloud_2eproto();  return *default_instance_;
}

PointCloud* PointCloud::default_instance_ = NULL;

PointCloud* PointCloud::New() const {
  return new PointCloud;
}

void PointCloud::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (has_source_serial()) {
      if (source_serial_ != &::google::protobuf::internal::kEmptyString) {
        source_serial_->clear();
      }
    }
    tick_ = 0u;
  }
  points_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool PointCloud::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required string source_serial = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_source_serial()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8String(
            this->source_serial().data(), this->source_serial().length(),
            ::google::protobuf::internal::WireFormat::PARSE);
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(16)) goto parse_tick;
        break;
      }
      
      // required uint32 tick = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_tick:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &tick_)));
          set_has_tick();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(26)) goto parse_points;
        break;
      }
      
      // repeated .RemoteKinect.Point points = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_points:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_points()));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(26)) goto parse_points;
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void PointCloud::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required string source_serial = 1;
  if (has_source_serial()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->source_serial().data(), this->source_serial().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    ::google::protobuf::internal::WireFormatLite::WriteString(
      1, this->source_serial(), output);
  }
  
  // required uint32 tick = 2;
  if (has_tick()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(2, this->tick(), output);
  }
  
  // repeated .RemoteKinect.Point points = 3;
  for (int i = 0; i < this->points_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      3, this->points(i), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* PointCloud::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required string source_serial = 1;
  if (has_source_serial()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8String(
      this->source_serial().data(), this->source_serial().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE);
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->source_serial(), target);
  }
  
  // required uint32 tick = 2;
  if (has_tick()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(2, this->tick(), target);
  }
  
  // repeated .RemoteKinect.Point points = 3;
  for (int i = 0; i < this->points_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        3, this->points(i), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int PointCloud::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required string source_serial = 1;
    if (has_source_serial()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->source_serial());
    }
    
    // required uint32 tick = 2;
    if (has_tick()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->tick());
    }
    
  }
  // repeated .RemoteKinect.Point points = 3;
  total_size += 1 * this->points_size();
  for (int i = 0; i < this->points_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->points(i));
  }
  
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void PointCloud::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const PointCloud* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const PointCloud*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void PointCloud::MergeFrom(const PointCloud& from) {
  GOOGLE_CHECK_NE(&from, this);
  points_.MergeFrom(from.points_);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_source_serial()) {
      set_source_serial(from.source_serial());
    }
    if (from.has_tick()) {
      set_tick(from.tick());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void PointCloud::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void PointCloud::CopyFrom(const PointCloud& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool PointCloud::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000003) != 0x00000003) return false;
  
  for (int i = 0; i < points_size(); i++) {
    if (!this->points(i).IsInitialized()) return false;
  }
  return true;
}

void PointCloud::Swap(PointCloud* other) {
  if (other != this) {
    std::swap(source_serial_, other->source_serial_);
    std::swap(tick_, other->tick_);
    points_.Swap(&other->points_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata PointCloud::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = PointCloud_descriptor_;
  metadata.reflection = PointCloud_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace RemoteKinect

// @@protoc_insertion_point(global_scope)
