"""Comprehensive example showing all available logger sinks.

This example demonstrates how to use console, file, WandB, and TensorBoard sinks
together in a realistic ML training scenario.
"""

import tempfile
from pathlib import Path
from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    MaxField,
    ListField,
    SumField,
    ConsoleSink,
    FileSink,
    # WandBSink,     # Uncomment if wandb is installed
    # TensorBoardSink # Uncomment if tensorboardX is installed
)

class MLTrainingLogger(LoggerBase):
    """Comprehensive logger for ML training with multiple metrics."""
    
    # Training progress
    epoch: Field[int]
    iteration: Field[int]
    
    # Loss metrics
    train_loss: MeanField
    val_loss: MeanField
    
    # Performance metrics  
    accuracy: MaxField[float]
    f1_score: MaxField[float]
    
    # Resource usage
    memory_mb: MaxField[float]
    training_time: SumField[float]
    
    # Events and messages
    events: ListField[str]

def main():
    """Demonstrate comprehensive multi-sink logging."""
    
    # Create temporary directories and files
    temp_dir = Path(tempfile.mkdtemp())
    log_file = temp_dir / "training.jsonl"
    # tb_dir = temp_dir / "tensorboard"  # Uncomment for TensorBoard
    
    print(f"🗂️  Working directory: {temp_dir}")
    print(f"📄 Log file: {log_file}")
    # print(f"📊 TensorBoard dir: {tb_dir}")  # Uncomment for TensorBoard
    
    # Set up sinks
    sinks = []
    
    # 1. Console sink - for immediate feedback
    console_sink = ConsoleSink(
        show_timestamp=True,
        show_logger_name=True,
        timestamp_format="%H:%M:%S"
    )
    sinks.append(console_sink)
    
    # 2. File sink - for persistent storage
    file_sink = FileSink(log_file, mode="w")
    sinks.append(file_sink)
    
    # 3. WandB sink - uncomment if wandb is available
    # try:
    #     wandb_sink = WandBSink(
    #         project="expedantic-demo",
    #         step_field="iteration",
    #         prefix="train_"
    #     )
    #     sinks.append(wandb_sink)
    #     print("✅ WandB sink enabled")
    # except ImportError:
    #     print("⚠️  WandB not available, skipping WandB sink")
    
    # 4. TensorBoard sink - uncomment if tensorboardX/torch is available  
    # try:
    #     tb_sink = TensorBoardSink(
    #         log_dir=tb_dir,
    #         step_field="iteration"
    #     )
    #     sinks.append(tb_sink)
    #     print("✅ TensorBoard sink enabled")
    # except ImportError:
    #     print("⚠️  TensorBoard not available, skipping TensorBoard sink")
    
    # Create logger with all sinks
    with MLTrainingLogger(name="ComprehensiveDemo", sinks=sinks) as logger:
        
        print("\n🚀 Starting comprehensive training demo...")
        print("="*60)
        
        # Simulate training loop
        iteration = 0
        
        for epoch in range(3):
            print(f"\n📈 Epoch {epoch + 1}/3")
            logger.epoch.log(epoch)
            
            # Training phase
            for batch in range(5):
                iteration += 1
                logger.iteration.log(iteration)
                
                # Simulate training metrics
                base_loss = 1.0 - (epoch * 0.3) - (batch * 0.02)
                train_loss = max(0.1, base_loss + (hash(str(iteration)) % 100) * 0.001)
                logger.train_loss.log(train_loss)
                
                # Simulate resource usage
                memory_usage = 512 + (iteration * 2) + (hash(str(iteration)) % 50)
                logger.memory_mb.log(memory_usage)
                
                training_time = 0.1 + (hash(str(iteration)) % 20) * 0.005
                logger.training_time.log(training_time)
                
                # Occasional events
                if iteration % 7 == 0:
                    logger.events.log(f"Checkpoint saved at iteration {iteration}")
            
            # Validation phase (once per epoch)
            val_loss = max(0.05, base_loss * 0.8 + (hash(str(epoch)) % 50) * 0.002)
            logger.val_loss.log(val_loss)
            
            accuracy = min(1.0, 0.6 + (epoch * 0.15) + (hash(str(epoch)) % 30) * 0.003)
            logger.accuracy.log(accuracy)
            
            f1 = min(1.0, accuracy * 0.95 + (hash(str(epoch)) % 20) * 0.002)
            logger.f1_score.log(f1)
            
            logger.events.log(f"Epoch {epoch} validation complete")
            
            # Flush sends to all configured sinks
            entry = logger.flush()
            
            print(f"   └─ Logged entry #{len(logger)} to {len(logger.sinks)} sinks")
        
        print(f"\n✅ Training complete! Generated {len(logger)} entries")
        print(f"📊 Data sent to {len(logger.sinks)} sinks:")
        for i, sink in enumerate(logger.sinks, 1):
            print(f"   {i}. {sink.__class__.__name__}")
    
    # Show some results
    print(f"\n📄 Sample file contents (first 2 lines):")
    try:
        with open(log_file) as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                print(f"   {i+1}: {line.strip()[:100]}...")
    except Exception as e:
        print(f"   Error reading file: {e}")
    
    # Print TensorBoard command if TB logs were created
    # if tb_dir.exists():
    #     print(f"\n📊 To view TensorBoard logs, run:")
    #     print(f"   tensorboard --logdir {tb_dir}")
    
    # Cleanup
    try:
        log_file.unlink()
        temp_dir.rmdir()
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

if __name__ == "__main__":
    main()