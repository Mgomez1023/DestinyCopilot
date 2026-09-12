export interface AuthStatus {
  configured: boolean;
  authenticated: boolean;
  message: string | null;
}

export interface CharacterSummary {
  character_id: string;
  class_name: string;
  race_name: string;
  gender_name: string;
  power: number;
  last_played: string | null;
  minutes_played_total: number;
  emblem_url: string | null;
  emblem_background_url: string | null;
  subclass: ItemSummary | null;
  equipped_gear: ItemSummary[];
  quests: QuestSummary[];
  milestones: MilestoneSummary[];
  progressions: ProgressionSummary[];
  available_activities: AvailableActivitySummary[];
  recent_activities: RecentActivitySummary[];
}

export interface ObjectiveSummary {
  objective_hash: number;
  name: string;
  description: string | null;
  progress: number | null;
  completion_value: number;
  progress_percent: number | null;
  complete: boolean;
  visible: boolean;
  activity_name: string | null;
  destination_name: string | null;
}

export interface ItemSummary {
  item_hash: number;
  instance_id: string | null;
  name: string;
  description: string | null;
  item_type: string;
  item_subtype: string | null;
  tier: string | null;
  icon_url: string | null;
  bucket_name: string | null;
  quantity: number;
  power: number | null;
  damage_type: string | null;
  is_equipped: boolean;
  is_locked: boolean;
  is_crafted: boolean;
  energy_capacity: number | null;
  energy_used: number | null;
  stats: Array<{ name: string; value: number }>;
  socketed_plugs: string[];
  socketed_plug_details: Array<{
    item_hash: number;
    name: string;
    description: string | null;
    item_type: string | null;
    category_identifier: string | null;
  }>;
  location: "equipped" | "character" | "vault" | "profile";
  character_id: string | null;
}

export interface QuestSummary {
  quest_hash: number;
  step_hash: number | null;
  name: string;
  step_name: string | null;
  description: string | null;
  icon_url: string | null;
  character_id: string;
  tracked: boolean;
  started: boolean;
  completed: boolean;
  redeemed: boolean;
  objectives: ObjectiveSummary[];
}

export interface MilestoneSummary {
  milestone_hash: number;
  name: string;
  description: string | null;
  character_id: string;
  start_date: string | null;
  end_date: string | null;
  activity_names: string[];
  quest_names: string[];
  objectives: ObjectiveSummary[];
}

export interface ProgressionSummary {
  progression_hash: number;
  faction_hash: number | null;
  name: string;
  description: string | null;
  scope: "profile" | "character" | "faction";
  character_id: string | null;
  level: number;
  level_cap: number;
  current_progress: number;
  progress_to_next_level: number;
  next_level_at: number;
  daily_progress: number;
  daily_limit: number;
  weekly_progress: number;
  weekly_limit: number;
  current_reset_count: number;
}

export interface AvailableActivitySummary {
  activity_hash: number;
  name: string;
  description: string | null;
  character_id: string;
  activity_type: string | null;
  destination: string | null;
  difficulty: string | null;
  display_level: number | null;
  recommended_power: number | null;
  is_new: boolean;
  can_lead: boolean;
  can_join: boolean;
  is_visible: boolean;
  is_completed: boolean;
  objectives: ObjectiveSummary[];
}

export interface RecentActivitySummary {
  activity_hash: number;
  name: string;
  description: string | null;
  character_id: string;
  period: string | null;
  instance_id: string | null;
  mode: number | null;
  activity_type: string | null;
  destination: string | null;
  difficulty: string | null;
  recommended_power: number | null;
  completed: boolean | null;
  duration_seconds: number | null;
}

export interface GuardianContext {
  bungie_display_name: string;
  membership_id: string;
  membership_type: number;
  platform_name: string;
  last_played: string | null;
  total_minutes_played: number;
  current_season_hash: number | null;
  current_season_name: string | null;
  characters: CharacterSummary[];
  inventory: {
    total_items: number;
    vault_items: number;
    character_items: number;
    unique_item_hashes: number;
    by_type: Record<string, number>;
    items: ItemSummary[];
    returned_items: number;
    truncated: boolean;
  };
  currencies: Array<{
    item_hash: number;
    name: string;
    description: string | null;
    quantity: number;
    max_stack_size: number | null;
    icon_url: string | null;
  }>;
  profile_progressions: ProgressionSummary[];
  collectibles: { total_visible: number; acquired: number };
  records: { total_visible: number; completed: number; near_completion: ObjectiveSummary[] };
  crafting: {
    total_visible: number;
    requirements_met: number;
    incomplete_pattern_names: string[];
  };
  data_availability: {
    fetched_at: string;
    available_components: string[];
    unavailable_components: Record<string, string>;
    notes: string[];
  };
  data_scope: string[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  message: string;
  source: "openai" | "local";
}
