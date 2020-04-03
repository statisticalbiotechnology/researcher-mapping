import { TestBed } from '@angular/core/testing';

import { D3PackedBubbleChartService } from './d3-packed-bubble-chart.service';

describe('D3PackedBubbleChartService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: D3PackedBubbleChartService = TestBed.get(D3PackedBubbleChartService);
    expect(service).toBeTruthy();
  });
});
