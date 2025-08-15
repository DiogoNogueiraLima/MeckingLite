import { Body, Controller, HttpException, Post } from '@nestjs/common';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';

const execAsync = promisify(exec);

type MoveRequest = { fen: string };
type MoveResponse = { 
  move?: string;
  game_over?: boolean;
  result?: string;
  winner?: string;
};

@Controller()
export class MoveController {
  @Post('move')
  async move(@Body() body: MoveRequest): Promise<MoveResponse> {
    if (!body?.fen) {
      throw new HttpException('Missing fen', 400);
    }

    try {
      // Caminho para o script Python (hardcoded para funcionar)
      const projectRoot = '/Users/Marcelo-Petroni/Documents/MeckingLite';
      const scriptPath = path.join(
        projectRoot,
        'model',
        'inference_service.py',
      );
      const pythonPath = 'python3';

      // Executar script Python passando o FEN
      const command = `cd "${projectRoot}" && ${pythonPath} "${scriptPath}" "${body.fen}"`;

      const { stdout, stderr } = await execAsync(command, {
        timeout: 30000, // 30 segundos timeout
        env: { ...process.env, PYTHONPATH: projectRoot },
      });

      if (stderr) {
        console.error('Python stderr:', stderr);
      }

      // Parse da resposta JSON
      const result = JSON.parse(stdout.trim());

      if (result.error) {
        throw new HttpException(result.error, 400);
      }

      // Verificar se é final de jogo
      if (result.game_over) {
        return {
          game_over: result.game_over,
          result: result.result,
          winner: result.winner,
        };
      }

      // Lance normal
      if (!result.move) {
        throw new HttpException('No move returned from model', 500);
      }

      return { move: result.move };
    } catch (err: any) {
      console.error('Model inference error:', err);

      if (err.code === 'ETIMEDOUT') {
        throw new HttpException('Model timeout', 504);
      }

      const message = err.message || 'Model inference failed';
      throw new HttpException(message, 500);
    }
  }
}
