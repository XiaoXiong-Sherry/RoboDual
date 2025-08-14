# WebSocket Evaluation Protocol

## Overview

This document describes the WebSocket communication protocol between the evaluation client and server for Calvin robot evaluation.

## Connection

- **Protocol**: WebSocket (ws:// or wss://)
- **Default Port**: 8765
- **URL Format**: `ws://host:port` or `wss://host:port`
- **Max Message Size**: 10MB
- **Ping Interval**: 30 seconds
- **Ping Timeout**: 10 seconds

## Message Format

All messages are JSON objects with a `type` field indicating the message type.

```json
{
  "type": "message_type",
  "...": "additional fields"
}
```

## Message Types

### 1. Connection Acknowledgment

**Direction**: Server → Client  
**Sent**: After client connects

```json
{
  "type": "connection_ack",
  "status": "connected",
  "client_id": "192.168.1.100:12345"
}
```

### 2. Ping/Pong (Heartbeat)

**Direction**: Client → Server → Client

**Ping Request**:
```json
{
  "type": "ping",
  "timestamp": 1640995200.123
}
```

**Pong Response**:
```json
{
  "type": "pong",
  "timestamp": 1640995200.123
}
```

### 3. Model Reset

**Direction**: Client → Server → Client  
**Purpose**: Reset model state for new evaluation sequence

**Request**:
```json
{
  "type": "reset"
}
```

**Response**:
```json
{
  "type": "reset_response",
  "status": "success"
}
```

### 4. Action Prediction

**Direction**: Client → Server → Client  
**Purpose**: Get action prediction from model

**Request**:
```json
{
  "type": "predict",
  "observation": {
    "rgb_obs": {
      "rgb_static": "iVBORw0KGgoAAAANSUhEUgAA...",  // base64 PNG
      "rgb_gripper": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    "depth_obs": {
      "depth_static": [[1.2, 1.3, ...], ...],      // 2D array
      "depth_gripper": [[0.8, 0.9, ...], ...]
    },
    "robot_obs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]  // 1D array
  },
  "instruction": "pick up the red block",
  "step": 15
}
```

**Success Response**:
```json
{
  "type": "predict_response",
  "status": "success",
  "action": [0.05, -0.02, 0.1, 0.0, 0.0, 0.0, 1.0],  // 7D action
  "step": 15
}
```

**Error Response**:
```json
{
  "type": "predict_response",
  "status": "error",
  "message": "Model inference failed: CUDA out of memory"
}
```

### 5. Error Messages

**Direction**: Server → Client  
**Purpose**: Communicate errors

```json
{
  "type": "error",
  "status": "error",
  "message": "Unknown message type: invalid_type"
}
```

## Data Serialization

### RGB Images
- **Format**: PNG images encoded as base64 strings
- **Color Space**: RGB
- **Typical Size**: ~200KB raw, ~270KB encoded

### Depth Images
- **Format**: 2D arrays of float values
- **Units**: Meters
- **Range**: 0.0 to 6.2 (static), 0.0 to 2.0 (gripper)

### Robot State
- **Format**: 1D array of 7 float values
- **Components**: [x, y, z, rx, ry, rz, gripper_state]
- **Units**: Position in meters, rotation in radians, gripper 0-1

### Actions
- **Format**: 1D array of 7 float values
- **Components**: [dx, dy, dz, drx, dry, drz, gripper_action]
- **Units**: Relative movements, gripper -1/1

## Communication Flow

```
Client                          Server
  |                               |
  |  WebSocket Connection         |
  |<=============================>|
  |                               |
  |  connection_ack               |
  |<------------------------------|
  |                               |
  |  reset                        |
  |------------------------------>|
  |  reset_response               |
  |<------------------------------|
  |                               |
  |  predict (step 0)             |
  |------------------------------>|
  |  predict_response             |
  |<------------------------------|
  |                               |
  |  predict (step 1)             |
  |------------------------------>|
  |  predict_response             |
  |<------------------------------|
  |                               |
  |  ... (continue for sequence)  |
  |                               |
  |  reset (new sequence)         |
  |------------------------------>|
  |  reset_response               |
  |<------------------------------|
```

## Error Handling

### Connection Errors
- **Reconnection**: Client should attempt automatic reconnection with exponential backoff
- **Timeout**: 30-second connection timeout
- **Max Retries**: 5 attempts before failing

### Message Errors
- **Invalid JSON**: Server responds with error message
- **Unknown Type**: Server responds with error message
- **Processing Error**: Server responds with error message and continues

### Performance Considerations

- **Concurrent Clients**: Server supports multiple simultaneous clients
- **Memory Usage**: ~500MB per loaded model
- **Throughput**: ~10-50 predictions/second depending on hardware
- **Latency**: ~50-200ms per prediction (local network)

## Usage Examples

### Starting WebSocket Server
```bash
python evaluation_server_ws.py \
  --generalist_path openvla7b \
  --specialist_path specialist_policy.pt \
  --host 0.0.0.0 \
  --port 8765
```

### Running WebSocket Client
```bash
python evaluate_calvin_websocket.py \
  --mode websocket-client \
  --server_url ws://192.168.1.100:8765
```

### WebSocket vs HTTP REST Comparison

| Feature | WebSocket | HTTP REST |
|---------|-----------|-----------|
| Connection | Persistent | Per-request |
| Overhead | Low | High |
| Latency | Lower | Higher |
| Complexity | Medium | Low |
| Debugging | Medium | Easy |
| Firewall | Good | Excellent |

## Security Considerations

- **Authentication**: Currently none implemented
- **Encryption**: Use wss:// for encrypted connections
- **Authorization**: Currently none implemented
- **Rate Limiting**: Currently none implemented

## Dependencies

### Server
```python
websockets>=10.0
torch
transformers
PIL
numpy
```

### Client
```python
websockets>=10.0
asyncio
PIL
numpy
requests  # for fallback HTTP mode
```