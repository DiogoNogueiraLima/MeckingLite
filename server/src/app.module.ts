import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MoveController } from './move.controller';

@Module({
  imports: [],
  controllers: [AppController, MoveController],
  providers: [AppService],
})
export class AppModule {}
